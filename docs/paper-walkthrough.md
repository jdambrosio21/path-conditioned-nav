# The Paper, End to End

A walkthrough of Haro et al., *Path-conditioned Reinforcement Learning-based Local
Planning for Long-Range Navigation* — what problem it solves, why the solution is
non-obvious, and how each piece maps onto this codebase.

---

## 1. The problem

Get a ground robot from A to B, where B is far away — across a building, through a
construction site — with only local sensing and an imperfect prior map.

Two classical answers, each with a fatal flaw:

**Global planner alone** (A\*/PRM/RRT over a prior map). Produces a geometrically valid
route, reasons about the whole environment, avoids dead ends. But it assumes the map is
accurate and current. Real maps are stale, built from noisy SLAM, missing the pallet
someone parked in the corridor an hour ago. And a planner cannot react — it replans, at
best, at a fraction of control rate.

**Learned local policy alone** (end-to-end RL on onboard sensing). Reactive, handles
clutter and dynamics beautifully, degrades gracefully. But it is *blind past sensor
range*. Faced with a U-shaped obstacle or a dead-end corridor, it walks in and gets
stuck, because nothing in its observation tells it the world beyond 10 m.

The two are complementary, so the standard hybrid is: **global planner emits a path,
local policy tracks it.**

## 2. Why the standard hybrid is not good enough

Tracking makes the local policy *subordinate* to the path. That produces two failure
modes that show up constantly on real robots:

- **The path is wrong.** Stale map, localization drift, a planner working from a bad
  cost map. The robot faithfully tracks the path straight into a wall, or into a region
  that no longer exists. The better it tracks, the worse it does.
- **The path is merely suboptimal.** The planner routes around the long way. The robot
  dutifully follows, wasting time, even when a shortcut is plainly visible in its own
  sensor data.

You can patch this — path-quality estimators, replanning triggers, switching logic
between "follow" and "explore" modes. All of it is hand-tuned, brittle, and adds
failure modes of its own.

## 3. The insight

**Give the policy the path as an *observation*. Reward it only for reaching the goal.**

That is the entire idea. Three consequences fall out:

- Because the path is an *observation*, the policy may use it or ignore it. It is
  information, not a constraint.
- Because there is **no path-following reward term**, nothing forces adherence.
- Because the goal reward is dense in *geodesic progress*, the policy has an objective
  of its own that is independent of the path.

Now think about what the optimal policy does. When the path is good, it encodes global
topology the robot cannot see — which side of the building to go around. Following it is
*also* the fastest route to the goal, so following is optimal, and the policy follows.
When the path is bad, following it costs reward, so the policy deviates.

**The mode switching is emergent.** No path-quality estimator, no confidence threshold,
no switching logic. One policy, one reward, and the behaviour falls out of the fact
that path-following was never what it was optimizing.

That is the contribution. Everything below is machinery to make it trainable.

---

## 4. The machinery

### 4.1 Path generation — the training distribution *is* the method

This is the part people underrate. **If you train only on optimal paths, you get a
tracker.** The policy learns `path == truth` because in training it always was. All the
robustness comes from training on a *mixture* of path qualities.

The paper builds a PRM (sample free-space nodes, connect k-nearest by collision-free
edges) and draws two grades of route from it:

- **A\*** on the roadmap → the optimal path.
- **Greedy Best-First Search under a biased heuristic** → structured suboptimality.
  GBFS ignores accumulated cost, so it is greedy by construction; multiplying the
  heuristic by a smooth random field makes it commit to a plausible-looking but
  needlessly long corridor.
- Waypoints are then perturbed by up to **1 m**, modelling registration error.

> **The design lesson.** The corruptions must be *plausible*. IID per-waypoint noise
> produces a zigzag no planner would ever emit — the policy would learn "zigzag ⇒
> ignore", a cue that does not exist in deployment. Smooth displacement models the real
> defect: a correct path registered against a misaligned map. **Your corruption model is
> a claim about how the upstream system fails.** Get it wrong and you train robustness to
> a fiction.

*Ours:* `planning.py` (`build_prm`, `astar`, `gbfs_biased`, `perturb`), plus the extra
degraded conditions the ablation needs in `PathQuality`.

### 4.2 Path encoding — attention over waypoints

15 waypoints go in; one context vector comes out. Two stages: **self-attention** across
the waypoint sequence, then **cross-attention** from the robot-state embedding into it.

Why not just concatenate 15 (x, y) pairs into the MLP?

- **Self-attention** lets waypoints see each other, so the encoder can represent path
  *shape* — a sharp turn 4 m ahead, a long straight, a doubling-back.
- **Cross-attention from the robot state** is the robot asking *"which part of this path
  matters to me right now?"* A waypoint 15 m out is irrelevant when you are 0.5 m from a
  wall. A flat MLP has to learn that gating with fixed per-position weights.
- **Masking** handles variable-length paths natively — including the zero-length case.

*Ours:* `models/path_encoder.py`. Note the paper's encoder is only **12,960 parameters**
— it is meant to be small. It summarizes a short sequence, not an image.

### 4.3 Asymmetric actor-critic — privileged learning

The actor sees only what a real robot could: local sensing, goal direction,
proprioception, and the **possibly corrupted** path. The critic additionally sees
privileged simulator state and the **true optimal path** — even on episodes where the
actor was handed a lie.

Why this is necessary, concretely: take a `WRONG_GOAL` episode. If the critic also only
saw the wrong path, it would value states by how good they look *relative to a path
leading somewhere else*. The advantage estimate would be garbage on exactly the episodes
that are supposed to teach robustness. Feeding the critic ground truth means the
advantage says *"this state is genuinely good or bad"*, so the actor learns the right
lesson from a corrupted-path episode instead of a confused one.

The general principle — **the critic is discarded at deployment, so it may cheat
freely** — is one of the most reusable ideas in robot learning.

*Ours:* `models/actor_critic.py`; the `priv` and `opt_path` observation channels in
`envs/torch_env.py`.

### 4.4 Reward design

- **Dense geodesic progress** toward the goal — the actual objective.
- **Terminal goal bonus.**
- **No path-following term.** This is the crux; adding one destroys the whole property.
- **Shortcut reward** — the paper's novel term, credit for opportunistic deviation.
- Regularization: action smoothness, collision, unsafe proximity, time.

On the shortcut term: pure goal-reaching *already* produces shortcutting in principle.
But the learning signal is weak, because leaving a path is a high-variance exploration
move that mostly ends in a collision before it pays off. The shortcut term sharpens
credit assignment — it fires at the exact moment the robot gains more true progress than
the path arclength it consumed.

> The paper does not give the functional form. Ours:
> `max(0, Δgeodesic − Δarclength)`, active only when a path exists. See
> `RewardConfig.shortcut` and `torch_env.step`.

### 4.5 Two-rate hierarchy

The navigation policy runs at **5 Hz** and emits velocity commands. A **frozen**
locomotion policy at **50 Hz** tracks them.

This decomposition by timescale is standard and worth internalizing: the navigation
policy never learns locomotion, the locomotion policy is reused across tasks, and the
navigation action space stays small and physically meaningful. It also means the
navigation policy is *portable across robots* — which is precisely why retargeting this
paper to a wheeled base was tractable at all.

*Ours:* `SUBSTEPS_PER_ACTION` in `config.py`; `_apply_action` in `envs/torch_env.py`;
the skid-steer inverse kinematics in `sim/mjcf.py` is the wheeled analogue of their
frozen locomotion controller.

### 4.6 Regularizers

The paper reports **Temporally Consistent Dropout** (mask held fixed for an episode
rather than redrawn per step, forcing robustness to a *persistent* missing feature
subset) and **Deep Mutual Learning** (two policies trained jointly with a KL coupling).

*Ours:* the first is implemented (`TemporallyConsistentDropout`). The second is not —
it doubles training cost for a regularization effect, and it is orthogonal to the claim.

---

## 5. What they measured

Training: 1046 parallel agents, 180 procedural arenas, 30×30 m for training and 50×50 m
for evaluation, 60 s episodes (120 s at evaluation), 39.5 h on one RTX 4090.

The experiment that matters is the **path-quality sweep**. The predicted signature:

- **Success rate stays roughly flat** as the reference path degrades → the policy never
  became dependent on guidance.
- **Efficiency improves when the path is good** → the policy is genuinely exploiting it.

Either one alone is a failure. Flat success with flat efficiency means the policy
learned to ignore the path entirely — you built an expensive goal-seeker. Improving
efficiency with collapsing success means you built a tracker. **The claim is that you
get both, and it is the absence of the path-following reward that buys it.**

*Ours:* `scripts/evaluate.py`.

---

## 6. Component map

| Paper component | This repo |
|---|---|
| Procedural 30×30 m arenas | `maps.py` |
| PRM + A\* optimal paths | `planning.py: build_prm, astar` |
| Biased-GBFS suboptimal paths | `planning.py: gbfs_biased` |
| ±1 m waypoint perturbation | `planning.py: perturb`, GPU version in `torch_env._smooth_offsets` |
| Depth camera (40×64, 105° FOV, 10 m) | `torch_env._cast_rays` — 64-ray 2-D analogue; see `docs/analytic-depth.md` |
| 15-waypoint attention encoder | `models/path_encoder.py` |
| Asymmetric actor-critic | `models/actor_critic.py` |
| Goal reward, no path-following term | `config.RewardConfig` |
| Shortcut reward | `torch_env.step` |
| PPO (rsl-rl) | `algorithms/ppo.py` |
| 5 Hz nav / 50 Hz low-level split | `config.SUBSTEPS_PER_ACTION` |
| Temporally Consistent Dropout | `models/path_encoder.py` |
| Path-quality ablation | `scripts/evaluate.py` |
| Deep Mutual Learning | *not implemented* |
| Isaac Sim | replaced — see README |

---

## 7. The three ideas worth stealing

Independent of this paper, these generalize:

1. **Supply information as observation, not as constraint.** Whenever you are tempted to
   reward imitation of some upstream module, ask whether you can instead feed that
   module's output as an observation and reward the *actual* objective. You get graceful
   degradation for free, and you stop inheriting the upstream module's bugs.

2. **Privileged critics.** The critic is thrown away at deployment. Anything the
   simulator knows can go into it. This is nearly free variance reduction, and it is the
   gateway to teacher–student distillation.

3. **Your corruption distribution is a modelling claim.** Robustness is only ever
   robustness *to the perturbations you trained on*. Designing those perturbations to
   match how the real upstream system actually fails is most of the work — and is far
   more important than the network architecture.
