# Results

Reproduction of Haro et al. (arXiv:2603.13888) on a wheeled robot, trained on an
M3 Max. 68.8M environment steps across two stages, ~3 h wall-clock.

All numbers are **deterministic** evaluation on **held-out** maps (different seed
from training), 200 episodes per condition.

---

## The headline experiment

| condition | success | efficiency | vs. path-blind baseline |
|---|---|---|---|
| `OPTIMAL` | **0.99** | **1.12** | +0.55 |
| `NOISY` | 0.99 | 1.14 | +0.55 |
| `SUBOPTIMAL` | 0.98 | 1.22 | +0.54 |
| `DETOURED` | 0.98 | 1.38 | +0.54 |
| `WRONG_GOAL` | **0.24** | 1.50 | **−0.20** |
| `NONE` | 0.43 | 1.86 | −0.01 |

Baseline: the same architecture trained **without ever seeing a path**, scoring
0.44 uniformly with efficiency 1.70–1.86.

Efficiency is distance travelled divided by the optimal route length; 1.0 is
perfect.

### What reproduced

**Path exploitation is large.** 0.43 → 0.99. Identical architecture, environment
and time budget; the only difference is being handed a reference path.

**The predicted signature is present.** Across `OPTIMAL → NOISY → SUBOPTIMAL →
DETOURED`, success stays flat (0.99, 0.99, 0.98, 0.98) while efficiency degrades
monotonically (1.12 → 1.14 → 1.22 → 1.38). This is the shape the paper's claim
requires, and it is not the shape either degenerate strategy produces:

- a **path tracker** would show collapsing success as the path degrades;
- a **path ignorer** would show flat efficiency, gaining nothing from good paths.

**No cost when the path is absent.** `NONE` = 0.43 against the 0.44 baseline, so
path conditioning did not damage path-free navigation.

### What did not reproduce

**`WRONG_GOAL` = 0.24, below the 0.43 no-path baseline.** Handed a well-formed
route to the wrong place, the policy follows it: 75% timeouts, driving to the
wrong goal and running out of time. The paper claims robustness here; we got the
tracker failure mode on precisely the condition designed to test it.

**Leading hypothesis, structural rather than a training failure.** The policy sees
15 waypoints at 1 m spacing — **15 m of path against routes averaging 60 m**. It
observes roughly 20% of the route, and the first 15 m of a wrong-goal path are
locally indistinguishable from a correct one. It may simply have no way to
perceive that the path ends somewhere else.

Two falsifiable tests:
1. Scale waypoint spacing to the remaining path length, so the window spans the
   whole route rather than a fixed 15 m.
2. Add the path's **endpoint** as an explicit observation, so it can be compared
   against the goal. One line, and a direct test of the hypothesis.

A secondary factor: `WRONG_GOAL` is 5% of training under a mixture we invented,
since the paper does not publish its sampling proportions — and it explicitly
notes that this choice "influences the learning dynamics."

---

## Sim-to-sim transfer

Trained on idealized kinematics, evaluated under MuJoCo rigid-body physics with
wheel/ground contact, finite traction, body roll and real collision response.

| condition | idealized | MuJoCo |
|---|---|---|
| `OPTIMAL` | 0.99 | **0.97** |
| `NOISY` | 0.99 | 0.88 |
| `SUBOPTIMAL` | 0.98 | 0.88 |
| `DETOURED` | 0.98 | 0.69 |
| `WRONG_GOAL` | 0.24 | 0.21 |
| `NONE` | 0.43 | 0.28 |

The policy transfers, and the path-quality ordering is preserved. The gap widens
as paths degrade, which is consistent: worse guidance means more manoeuvring, and
manoeuvring is where the two dynamics models differ most.

This experiment is not in the paper. It is available here only because the two
backends share every line of observation, reward and termination logic and differ
solely in `_apply_action`.

---

## Benchmark validity

The result above is only meaningful because the environment is hard enough to
measure it. Three earlier versions were not, and each looked fine until measured
properly.

| environment | path-blind policy | headroom |
|---|---|---|
| 30 m scattered obstacles | **0.97** | none |
| 30 m maze | **0.97** | none |
| **45 m maze, 90 s budget** | **0.44** | 55 points |

Scripted controllers, no learning, on the final environment:

| controller | success |
|---|---|
| greedy (ignores the path) | 0.4% |
| pure pursuit (follows it) | 42.1% |

A 105× ratio, against 4.1× in the original open arena.

**The check that matters** is the first table, not the second. A *scripted* greedy
controller failed at 0.4% on the 30 m maze while a *learned* path-blind policy
scored 0.97 on the same maps — validating difficulty against a weak baseline is
how three successive environments passed inspection while measuring nothing.

`tools/check_difficulty.py` runs the scripted version in two minutes.
`scripts/evaluate.py` on a path-blind checkpoint runs the real one in thirty.

---

## Throughput

M3 Max, 14 CPU / 30 GPU cores, 36 GB.

| | steps/s |
|---|---|
| Training, 4096 envs, SRU, 45 m mazes (MPS) | 7,300 |
| Training, feedforward (same setting) | 8,190 |

Recurrence costs ~5%, not the 3× predicted: the sequential GRU loop is small next
to the attention encoders and trunk.

Physics backends measured on the same machine:

| | steps/s |
|---|---|
| C MuJoCo, `mujoco.rollout`, 14 threads | 3,480,000 |
| C MuJoCo, single thread | 107,000 |
| MJX on `jax-mps` Metal (B=32768, async) | 47,800 |
| MJX on JAX CPU (B=2048) | 14,700 |

MJX-on-Metal plateaus from B=4096 — dispatch-bound, not compute-bound. The GPU
itself is fine (8.24 TFLOP/s on a 4096³ matmul against 1.00 on CPU); it just wants
large kernels, which a MuJoCo step is not. Hence the split used throughout:
closed-form dynamics and ray casting batched on the GPU, gradients on the GPU,
MuJoCo reserved for evaluation.

---

## Model

Actor — 154,564 parameters, the part that would deploy:

| component | params |
|---|---|
| State encoder (74-D → 64) | 8,960 |
| SRU memory (1 layer, h=64) | 28,928 |
| Path encoder (self-attn → cross-attn, 15 waypoints) | 50,496 |
| Trunk (256, 128) → 2 | 66,178 |

Critic: 171,073 parameters, discarded at deployment.

The paper's actor is 1.76M, almost all of the difference being a depth CNN we do
not have — their observation is a 2636-D flattened depth map against our 74-D
vector.
