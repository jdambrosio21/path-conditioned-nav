# Reading Guide — How to Get the Engineering, Not Just the Theory

You said you're comfortable with the theory. This doc is about the other half: the
implementation knowledge that doesn't appear in papers because it isn't publishable.

The organizing principle: **read only code you can run and verify within a few minutes
of reading it.** Passive reading of an RL codebase teaches almost nothing, because
everything in RL fails silently — the shapes are right, no exception is raised, the loss
goes down, and the policy is garbage. You build intuition by predicting a behaviour,
running it, and being wrong.

---

## Part 1 — Reading order

### 0. `tests/` first (~20 min)

Counterintuitive but correct. The tests are the **invariants** — the things that must be
true, stated compactly, without the surrounding machinery. `tests/test_env.py` tells you
what an observation is supposed to contain and what "no path" means before you read a
line of the env.

Read all three files. You will not understand every assertion yet. That's fine — you're
building a checklist of "things that apparently matter", which is exactly the frame you
want when you hit the implementation.

### 1. `config.py` (~15 min)

The vocabulary. Every constant with a comment explaining *why* it has that value.

**Ask as you read:** which of these are from the paper, which are properties of the
robot, and which are free parameters I chose? (Answer: roughly a third each. Knowing
which is which is knowing where the risk is.)

### 2. `maps.py` (~45 min) — pure numpy, no RL

Occupancy grids, obstacle inflation, connected components, Dijkstra distance fields.
Nothing here depends on learning, so it's fully verifiable in isolation.

**Key idea:** *inflate the obstacles by the robot radius and the robot becomes a point.*
This single trick makes planning, collision checking, and the geodesic field all trivial.
It recurs everywhere in robotics — configuration-space thinking.

> **Exercise 1.** Write a ~20-line matplotlib script: render one map's obstacles, overlay
> `free` as a mask, and plot `dist[0]` as a heatmap. You will *see* whether inflation and
> connectivity are correct. Ten minutes, and it makes the rest of the codebase concrete.
> Do this before reading anything else.

### 3. `planning.py` (~45 min) — pure algorithms

A\*, GBFS, resampling, perturbation.

**Ask:** why is the path resampled to *uniform arclength spacing* and zero-padded to a
fixed length? (Because it converts "project the robot onto its path" — normally a
per-segment geometric solve — into a single batched `argmin`. Data layout chosen to make
the hot path a tensor op. That's the whole game in batched simulation.)

> **Exercise 2.** Plot A\* and biased-GBFS routes for the same start/goal on one map.
> Look at how GBFS commits to a wrong corridor. Then set `bias_strength=0` and watch the
> suboptimality become boring and unstructured. That parameter is the difference between
> a realistic corruption and a useless one.

### 4. `envs/torch_env.py` (~2–3 h) — the heart

Don't read top to bottom. Read in this order:

1. **`_allocate_state`** — the data layout. Every tensor, its shape, its meaning. Write
   the shapes down. Everything else is operations on these.
2. **`_pack_maps_to_device`** — how a list of Python objects becomes indexable tensors.
3. **`_reset_envs`** — the trickiest function in the file. Note it operates on a *subset*
   of environments.
4. **`_cast_rays`** — the geometry, and the clearest example of broadcasting to
   `(envs, rays, obstacles)`.
5. **`_build_observation`** — where the actor/critic asymmetry becomes concrete.
6. **`step`** — reward assembly and termination.

> **Exercise 3.** Instantiate an env with `num_envs=1`, print the full observation, and
> **verify the goal bearing by hand** from `position`, `heading` and `goal_position`.
> Frame conventions are the #1 source of silent bugs in robotics code, and the only cure
> is to check one by hand at least once.

### 5. `models/path_encoder.py` (~45 min)

Small file, disproportionate subtlety. Focus entirely on the masking.

**Ask:** what happens if you delete the `attendable[:, 0] |= ~valid.any(dim=1)` line?
(Softmax over an all-`-inf` row → NaN → every gradient in the network becomes NaN → the
run dies with no useful error.) 10% of training episodes hit this path.

### 6. `models/actor_critic.py` (~30 min)

Read `action_mean` and `value` side by side. The asymmetry is literally which tensors
each one is handed.

### 7. `algorithms/ppo.py` (~1.5 h)

Buffer layout `(steps, envs, ...)`, GAE, the clipped surrogate, the KL-adaptive LR.

**Ask:** why does `compute_returns` multiply by `not_done`? (Because the environment
auto-resets — the observation at `t+1` belongs to a *new episode*. Without that mask you
bootstrap the value of an unrelated state across the episode boundary.) This is the most
common bug in vectorized-env RL, and it is invisible: the code runs, the loss decreases,
performance is just quietly worse.

### 8. `algorithms/runner.py` (~45 min)

The glue — and where the subtle bugs live, because it owns the *contract* between
rollout and update. Two bugs found during this project both lived here or in that
contract. See Part 2, item 6.

### 9. `envs/mujoco_env.py` + `sim/` (~1 h)

Note the structure: it replaces **only** `_apply_action` and inherits everything else.
That is what makes the sim-to-sim comparison mean something — the two backends cannot
differ in anything but dynamics, because they *share the code* for everything else.

---

## Part 2 — The transferable engineering patterns

The list I'd actually want someone to walk away with. Each is in the code; go find it.

**1. Padding with sentinels instead of masks.**
`_pack_maps_to_device` pads obstacle slots with `radius = -1`, so padding rows fail every
intersection and collision test *automatically*. No mask tensor, no branch. Find the
value that makes the math self-masking.

**2. `+inf` before `argmin`.**
`_project_onto_path` sets invalid vertices to `inf` so the `argmin` can't select them.
Branchless, batched, and obviously correct.

**3. Precompute everything that isn't per-step.**
Geodesic fields, ray bearings, roadmaps, the whole path library. Ask of every line in a
hot loop: *does this change between steps?*

**4. Move the device boundary, don't optimize around it.**
The path library exists because episode resets were doing CPU graph search inside a GPU
training loop. The fix wasn't faster A\* — it was making resets never touch the host.
When something is slow, first ask whether it should be happening there at all.

**5. Auto-reset semantics.**
The env resets internally, so `obs` returned from `step` may belong to a new episode.
Every consumer must know this. (See PPO's `not_done`.)

**6. Rollout/update consistency — the invariant that bit twice.**
*Anything stochastic in the policy must be exactly replayable at update time.* PPO's
importance ratio assumes you are re-scoring actions under *the same network* that
produced them. Both bugs found in this project were violations:
   - the dropout mask wasn't recorded in the buffer;
   - the attention stack had vanilla dropout, redrawing every forward.

Neither crashes. Neither shows up in a shape check. Both silently corrupt the gradient.
Whenever you add stochasticity to a policy, ask immediately: *is this replayable?*

**7. Estimator choice is an engineering decision.**
The ratio-based KL estimator is heavy-tailed — a few tail samples dominate the mean and
report KL = 0.49 while `clip_frac` says 0.02. Driving a controller with that is how the
first training run destroyed its own learning rate. The analytic Gaussian KL depends only
on distribution parameters: bounded, noise-free. **When two quantities that should agree
disagree, one of them is lying — find out which before tuning anything.**

**8. Feedback loops need gentle gains.**
The adaptive LR is a controller. A 1.5× step per iteration traverses the entire allowed
range in ~10 iterations, so one noisy reading strands it at a bound permanently. Dropped
to 1.1×. If you have a loop that adjusts a hyperparameter, you have a control system —
think about its stability.

**9. Build an independent oracle.**
The MuJoCo backend checks the torch env. The 2-D raycast will check the depth image. The
scripted pure-pursuit controller checked the whole environment *before any training*.
Structuring work so you always have something to check against is most of what separates
a result from a plausible-looking bug.

**10. A benchmark can measure nothing, and look fine.**
Three successive environments were solved at 0.97 by a policy that never saw a path.
Each passed a scripted-baseline check first -- a hand-written greedy controller scored
0.4% on the same maps -- because a *learned* policy with a laser scan is far stronger
than a scripted one. **Validate difficulty against the strongest baseline you have,
not the most convenient.** The real check is thirty minutes: train a policy without
the feature, evaluate deterministically, and require that it fails.

**11. Training curves understate policies.**
The maze run logged 0.60 while deterministic evaluation of that same checkpoint gave
0.97 -- exploration noise colliding in narrow corridors. Several "it is stalling"
readings were wrong by 35 points. Judge by evaluation, not the curve.

**12. Cross-component assumptions drift silently.**
The MuJoCo robot had a 0.569 m envelope while maps, roadmap and collision all modelled
a 0.350 m disc. It physically could not fit through gaps the planner called clear. Any
constant shared across module boundaries deserves a test that pins the two together.

**13. A masked bug can look like a working system.**
That geometry error was invisible while a *separate* control bug stopped the robot from
turning -- a crawling robot never notices it is too wide. Fixing the controller took
collisions from 71% to 100%, which reads as a regression and is actually progress.
When a fix makes things worse, suspect it unmasked something.

**14. Budget memory before writing code.**
`(envs × rays × obstacles)` — compute it first. It tells you the design (cull + chunk)
before you've written anything. See `docs/analytic-depth.md` §4.

---

## Part 3 — Break it on purpose

This is the highest-value section. For each: **write down the failure signature you
predict, then run it.** Being wrong is the point; that gap is the intuition you're
building.

| # | Break this | Predict, then check |
|---|---|---|
| 1 | Delete `- ROBOT_RADIUS_M` from `_clearance` | Collisions nearly vanish, success looks *great*. Lesson: a metric that improves because you broke its definition. |
| 2 | Remove the `in_range` masking in `_project_onto_path` | Robot projects onto padding vertices at the origin; arclength garbage; shortcut reward fires at random. |
| 3 | Normalize advantages per-minibatch instead of per-batch | Subtle: higher gradient variance, slower learning. Hard to see — which is why the whole-batch choice is deliberate. |
| 4 | Revert fix 6 (use the *current* dropout mask in `evaluate`) | KL and `clip_frac` climb; learning degrades but nothing errors. |
| 5 | `RewardConfig.shortcut = 0` | Success ~unchanged; efficiency on `SUBOPTIMAL` degrades. Isolates the paper's novel term. |
| 6 | `RewardConfig.progress = 0` (goal bonus only) | Barely learns at all. This is why dense shaping exists — a visceral lesson in sparse reward. |
| 7 | Feed the critic `path` instead of `opt_path` | `WRONG_GOAL` and `DETOURED` performance degrades specifically. Isolates why asymmetry matters. |
| 8 | Set `bias_strength = 0` in `gbfs_biased` | Suboptimal paths become unstructured; robustness transfers worse. |

Run these on a small config (`--num-envs 256 --num-maps 8 --device cpu`) so a cycle is
minutes, not hours.

---

## Part 4 — How to debug RL specifically

RL's defining pathology: **the only symptom is "it didn't learn."** One symptom, fifty
causes. So you need a fixed ladder, cheapest first, and the discipline to not skip steps.

1. **Does a scripted expert solve it?** If hand-written pure-pursuit can't, the *task or
   reward* is broken, not the algorithm. (We ran this before any training: 68% on
   `OPTIMAL` vs 21% on `NONE` — proof the env rewards path use.)
2. **Are random-policy statistics sane?** Episode lengths, termination reasons, reward
   component magnitudes. Print them.
3. **Can it overfit one map, one start/goal pair?** If it can't memorize a single
   episode, nothing downstream matters. Fastest high-signal test in RL.
4. **Are shapes, masks, and frames right?** This is what the test suite is for.
5. **Is the reward scale sane?** Log each component separately. One term with a 100×
   larger magnitude silently defines your objective.
6. **Is the value function learning?** `value_loss` should fall. If it's flat, the critic
   sees nothing predictive and every advantage is noise.
7. **Are the PPO diagnostics healthy?** `kl` ~ target, `clip_frac` ~ 0.05–0.2, entropy
   falling slowly, `action_std` not collapsing or exploding.
8. **Only now:** hyperparameters.

Most people start at 8. The ordering *is* the skill.

**And: instrument before you guess.** When training here ran at 3,676 steps/s, the
plausible story was "MPS is slow." Profiling said rollout 2.7 s, update 21.5 s — the
bottleneck was an oversized network, and reading the paper's parameter count fixed it.
The obvious explanation was wrong, and one measurement was worth an afternoon of tuning.

---

## Part 5 — A four-week plan

1. **Week 1 — comprehension.** Reading order above, plus Exercises 1–3. Deliverable: a
   diagram *you* draw of one environment step, every tensor and its shape.
2. **Week 2 — destruction.** Part 3, all eight. Deliverable: a note per experiment,
   prediction vs. actual.
3. **Week 3 — construction.** Implement analytic depth (`docs/analytic-depth.md`),
   steps 1–6. Pure geometry, fully testable, no RL.
4. **Week 4 — experiment.** Pits, retrain, run the comparison. Deliverable: a plot and an
   honest paragraph on whether the hypothesis held.

Then: **teacher–student privileged distillation** (Lee et al. 2020; Miki et al. 2022 —
same lab as this paper). The asymmetric critic here is halfway to it. That is the single
highest-value next project, and it reuses this entire codebase.
