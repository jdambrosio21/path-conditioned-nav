# Analytic Depth Rendering — Design Doc

**Goal:** produce a real 40×64 depth image, batched across thousands of environments,
on Apple Silicon — with **no renderer, no rasterizer, no graphics API**. Pure algebra
on tensors.

This is the thing MJX cannot do on this machine (no batched renderer; `madrona-mjx` is
CUDA-only). We sidestep it rather than fight it.

**Status:** camera model, ray-cylinder geometry, ground plane, arena walls, broad-phase
culling and ray chunking are implemented in `src/pcnav/sim/depth.py` and covered by
`tests/test_depth.py` (validated three ways: hand-computed single rays, agreement with
the existing 2-D caster on horizontal rays, and invariance to chunking). Not yet done:
ray-oriented-box for maze walls (§6b), the pits terrain, and the CNN encoder — so depth
is not yet wired into training.

---

## 1. The core idea

A depth camera is not a mysterious sensor. It is **a grid of rays with a distance
reported per ray**.

Real-time graphics renders by *rasterization*: project triangles forward onto the image
plane and keep the nearest via a z-buffer. That needs a graphics pipeline. But you can
invert the problem — for each pixel, shoot a ray *backwards* into the scene and solve
for the nearest intersection. If every object in the scene is an **analytic primitive**
(sphere, plane, cylinder, box), that intersection has a closed-form solution.

No pipeline, no z-buffer, no drivers. Just a quadratic per (ray, object) pair, and a
`min` reduction. All of it is dense tensor arithmetic, which is exactly what a GPU is
happiest doing — and unlike MJX's hundreds of tiny branchy kernels, this is **a handful
of huge fused kernels**, the shape Metal actually runs well (recall: 8.24 TFLOP/s on
one big matmul vs 48k steps/s on MJX).

We already do this in 2-D. `torch_env._cast_rays` casts 64 horizontal rays against
circles. Going to a depth image is *the same math in 3-D*, plus a camera model.

---

## 2. Camera model (pinhole)

Given image width `W=64`, height `H=40`, horizontal FOV `105°`, vertical FOV `78°`
(the paper's ZED X parameters).

Focal lengths in pixels:

```
f_x = (W / 2) / tan(hfov / 2)
f_y = (H / 2) / tan(vfov / 2)
```

For pixel `(u, v)`, sampling at the **pixel centre** (`+0.5` — forgetting this puts a
half-pixel skew in your whole image):

```
x_n = (2 * (u + 0.5) / W - 1) * tan(hfov / 2)
y_n = (2 * (v + 0.5) / H - 1) * tan(vfov / 2)
```

In the camera frame (x forward, y left, z up):

```
d_cam = normalize( [1, -x_n, -y_n] )
```

Then rotate into the world by the robot's yaw and the camera's fixed mount pitch:

```
d_world = R_z(yaw) @ R_y(mount_pitch) @ d_cam
o_world = robot_position + camera_mount_offset
```

Precompute `d_cam` **once** at startup — it is constant, shape `(H*W, 3)`. Only the
rotation is per-step.

### Depth vs. range — the classic trap

Ray-marching gives you `t`, the **Euclidean range** along the ray. Real depth cameras
(and MuJoCo's depth buffer) report **z-depth**: the distance along the optical axis.

```
depth = t * (d_cam · optical_axis) = t * d_cam[0]      # in our x-forward convention
```

They differ by up to `1/cos(52.5°) ≈ 1.64` at the image corners. If you train on range
and deploy against z-depth, the policy sees a systematically warped scene at the edges,
which is exactly where obstacle avoidance matters. **Store z-depth.**

---

## 3. Ray–finite-cylinder intersection

Our obstacles are vertical cylinders: axis at `(cx, cy)`, radius `r`, spanning
`z ∈ [z0, z1]`.

Ray: `p(t) = o + t·d`, with `t > 0` and `|d| = 1`.

### Side surface

Substitute into the circle equation in XY. With `fx = ox - cx`, `fy = oy - cy`:

```
(fx + t·dx)² + (fy + t·dy)² = r²
```

Expand to `a·t² + 2b·t + c = 0`:

```
a    = dx² + dy²
b    = fx·dx + fy·dy
c    = fx² + fy² - r²
disc = b² - a·c
t    = (-b - sqrt(disc)) / a        # near root
```

**This is the key generalization from our 2-D code.** In 2-D, rays were unit vectors in
the plane, so `a = 1` and it dropped out. In 3-D the ray tilts, so its XY projection is
*not* unit length and `a = dx² + dy² < 1`. Omitting `a` is the single most common bug
here — it produces depth that is subtly too short, worst for steeply tilted rays, and it
will *look* plausible.

Guard `a ≈ 0` (a ray pointing straight up or down never hits the side surface).

Then the **height test** — this is what makes the cylinder finite:

```
z_hit = oz + t·dz
valid = disc > 0  and  t > 0  and  z0 ≤ z_hit ≤ z1
```

If the near root fails the height test, the far root may still be valid (the ray entered
through the top and exits the side).

### Top cap

A disc at `z = z1`:

```
t_cap = (z1 - oz) / dz                              # guard dz ≈ 0
hit   = (ox + t·dx - cx)² + (oy + t·dy - cy)² ≤ r²
```

The bottom cap is almost always occluded by the ground; skip it.

### Ground plane

```
t_ground = -oz / dz          for dz < 0
```

Downward rays always hit. **This matters more than it looks** — ground returns fill the
lower half of a depth image, and they are how a depth camera perceives terrain shape.
It is also what makes *negative* obstacles visible (see §6).

### Walls

Axis-aligned boxes via the slab method (3-D version of what `_cast_rays` already does).

### Combine

```
depth = min over {cylinder sides, caps, ground, walls}, clipped to max_range
```

---

## 4. The memory problem, and how real renderers solve it

Naive broadcast shape is `(envs, rays, obstacles)`:

| envs | rays | obstacles | elements | float32 per tensor |
|---|---|---|---|---|
| 4096 | 2560 | 45 | 472 M | **1.9 GB** |
| 1024 | 2560 | 45 | 118 M | 472 MB |

And the quadratic needs `b`, `c`, `disc`, `t`, plus masks — call it 6× that. The naive
version does not fit. Two standard fixes, both worth internalizing:

### (a) Broad-phase culling

Most obstacles are behind the robot or beyond sensor range. Before casting, rank
obstacles per environment and keep only the nearest `K = 12`:

```python
surface_distance = ‖obstacle_center - robot_xy‖ - radius   # subtract r, see below
in_range = surface_distance < max_range
score    = where(in_range & in_front, surface_distance, +inf)
keep     = score.topk(K, largest=False).indices
```

**Rank by surface distance, not centre distance.** A large-radius obstacle can be
farther by centre yet closer by surface — cull by centre and you will drop the very
obstacle about to be hit. Classic bug.

This is exactly what production renderers do (frustum culling, BVH traversal). You are
implementing the same idea in three lines.

`45 → 12` obstacles is a **3.75× memory reduction.**

### (b) Chunk over rays

Process rays in blocks of e.g. 640 and loop 4 times. Wall-clock cost is near zero
(the GPU is saturated either way), memory drops 4×.

Combined: `1024 envs × 640 rays × 12 obstacles = 7.9 M` elements = **31 MB per tensor**.
Comfortable.

### Budget the whole thing before writing code

That table above should be the *first* thing you compute on any batched-perception
feature. It tells you the design (cull + chunk) before you have written a line.

---

## 5. Downstream consequences

Switching the observation from a 64-vector to a 40×64 image is not a drop-in.

**The trunk needs a CNN.** Feeding 2560 raw floats into an MLP is wasteful and ignores
spatial structure. A small encoder:

```
conv(1→16, k3 s2) → conv(16→32, k3 s2) → conv(32→64, k3 s2) → flatten → linear(→128)
40×64 → 20×32 → 10×16 → 5×8
```

≈ 50 k parameters. **This is why the paper's actor is 1.76 M parameters** — the depth
encoder dominates. Our current actor is 126 k total.

**Throughput will drop.** Expect the env step to go from ~0.03 s to something meaningful
for the first time — perception becomes comparable to the gradient step rather than
free. Measure before and after; do not guess.

**The observation normalization changes.** Depth in metres, clipped to 10, divided by
10 — and decide explicitly what "no return" means (sky, beyond range). Convention:
clamp to `max_range`, which reads as "free space out to the horizon."

---

## 6. The experiment that justifies the whole exercise

Adding depth is only worth it if there is something a 2-D horizontal scan **provably
cannot perceive**. Height variation alone is not enough — for a wheeled robot on flat
ground with vertical cylinders, a single horizontal slice is already a *complete*
description of the scene. Extra rows would be redundant pixels, and any "improvement"
you measured would be noise.

The clean case is **negative obstacles — pits and drop-offs.**

A horizontal ray at chassis height passes straight over a hole and reports whatever is
beyond it. The hole is *invisible*, in principle, not just in practice. A depth camera
sees it immediately: the ground-plane returns in the lower image rows jump to a longer
range (or to no return at all) exactly where the floor falls away.

So the terrain gets one addition: circular pits. Driving into one terminates the episode
as a failure, same as a collision.

**The hypothesis is then falsifiable and sharp:**

| Policy | Obstacles only | Obstacles + pits |
|---|---|---|
| 2-D raycast | ~equal | **fails — cannot see pits** |
| 40×64 depth | ~equal | succeeds |

If the depth policy does *not* beat the raycast policy on pit terrain, something is
wrong with the implementation — the raycast policy is receiving information it should
not have. That is a real experiment with a real prediction, not a demo.

---

## 6b. Outstanding: oriented-box walls

The environment now contains **trap structures** — U-shaped pockets, gapped
barriers, dead-end corridors — built from oriented boxes, because scattered convex
obstacles left the task solvable without a reference path at all (see
`docs/reading-guide.md`). The 2-D scan casts against them; the depth module does
**not** yet.

Ray–oriented-box is the slab method in the box's local frame, and the 2-D version
already exists in `torch_env._cast_rays_at_walls` — the 3-D extension adds a third
slab for height. Do this before any depth-vs-scan comparison, or the comparison
measures the missing primitive rather than the thing under test.

## 7. Implementation order

Build it so that every stage is verifiable before the next one depends on it.

1. **`sim/depth.py` — camera model.** Precompute `d_cam`. Unit-test: centre pixel points
   straight ahead; corner pixels sit at the expected FOV half-angles.
2. **Ray–cylinder intersection, single ray, numpy.** Test against hand-computed cases:
   head-on hit, tangent grazing, ray passing above the cylinder top, ray starting inside.
3. **Batch it.** Test that the horizontal centre row of the depth image **matches the
   existing 2-D `_cast_rays` output** for full-height cylinders. This is the single best
   test available — you have a known-good implementation to check against.
4. **Add ground plane and walls.** Test: with no obstacles, the depth image is a smooth
   gradient (near at the bottom, `max_range` at the horizon).
5. **Cull + chunk.** Test: output is *bitwise identical* to the unculled version on a
   scene with fewer than `K` obstacles.
6. **Cross-validate against MuJoCo.** Render the same scene with `mujoco.Renderer`'s
   depth buffer and compare. Agreement to a few centimetres means the math is right.
   *This is the payoff of having built the MuJoCo backend* — an independent oracle.
7. **Pits:** terrain generation, termination logic, and the negative-obstacle geometry in
   the depth cast.
8. **CNN encoder**, observation plumbing, retrain, run the §6 comparison.

Steps 1–6 are pure geometry and fully testable without any RL. Do not start training
until step 6 agrees with MuJoCo.

---

## 8. Why this is worth your time

The transferable skills here, in rough order of value:

- **Analytic/differentiable rendering.** The same technique underlies NeRF-era
  differentiable rendering, signed-distance-field robotics, and GPU collision checking.
  Once you see "rendering is just a batched root-find," a lot of graphics demystifies.
- **Memory budgeting for batched perception.** The `(envs, rays, objects)` explosion and
  the cull-and-chunk response is the single most reusable pattern in massively parallel
  simulation.
- **Building an independent oracle.** We can check analytic depth against MuJoCo's
  renderer. Structuring work so you *have* something to check against is most of what
  separates a result from a plausible-looking bug.
- **Designing a falsifiable experiment.** §6 is the real lesson: don't add a feature and
  report that the number went up. Find the case where the feature is *provably* necessary
  and predict the outcome in advance.

### Where this leads next

The natural follow-on in this literature is **teacher–student privileged distillation**
— the actual standard for legged/wheeled navigation (Lee et al. 2020, Miki et al. 2022,
both from the same lab as this paper). Our asymmetric critic is already halfway there:
the critic sees privileged state. The full method trains a *teacher policy* with
privileged access, then distills it into a *student* restricted to onboard sensing.

That is the single highest-value thing to build after this project, and it reuses
everything here: privileged observations, procedural terrain, the batched env.
