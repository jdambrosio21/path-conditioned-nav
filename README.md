# Path-Conditioned RL Local Planning — Wheeled Robot, Apple Silicon

A reimplementation of **"Path-conditioned Reinforcement Learning-based Local Planning
for Long-Range Navigation"** (Haro, Richter, Yang, Cadena, Hutter — [arXiv:2603.13888](https://arxiv.org/abs/2603.13888)),
retargeted from an Isaac Sim / quadruped setup to a **wheeled robot that trains on an M3 Max**.

The paper's claim, in one sentence: *give a navigation policy a reference path as an
**observation** rather than something it is rewarded for following, and it learns to
exploit good paths while staying robust to misleading ones.* That claim, and the
machinery behind it, is what this repository reproduces.

---

## Why this is not a line-by-line port

The paper trains in **Isaac Sim / Isaac Lab** on an **RTX 4090** (39.5 h), with a
Unitree B2W consuming **40×64 depth images** from 1046 parallel environments. None of
that runs on Apple Silicon:

- Isaac Sim/Lab is CUDA + Linux/Windows only.
- MJX has **no batched renderer**; `madrona-mjx` is CUDA-only. Batched depth imagery
  is unavailable on this hardware at any batch size, on any backend.

So the substrate was swapped and the contribution kept.

| | Paper | This repo | Rationale |
|---|---|---|---|
| Simulator (training) | Isaac Sim / Isaac Lab | Vectorized PyTorch env | Isaac is CUDA-only |
| Simulator (eval/viz) | — | **MuJoCo** | Real contact physics + viewer |
| Robot | Unitree B2W (wheeled quadruped) | Wheeled skid-steer base | Requested platform |
| Action | `(v_x, v_y, ω)` @ 5 Hz | `(v_x, ω)` @ 5 Hz | Wheeled base cannot strafe |
| Perception | 40×64 depth, 105° FOV, 10 m | 64-ray scan, **105° FOV, 10 m** | No batched renderer on Mac |
| Path generation | PRM + A\*, biased GBFS, ±1 m noise | **same** | Core method |
| Path encoder | self-attn → cross-attn, 15 waypoints | **same** | Core method |
| Reward | goal-reaching + shortcut, **no path-following term** | **same** | Core claim |
| Learning | asymmetric-critic PPO (rsl-rl) | asymmetric-critic PPO | Core method |

**Deviations worth knowing about**, beyond the table:

- `DETOURED` reference paths are "a biased-GBFS route made worse by a smooth
  displacement" rather than "a route forced through a random via point". Both yield a
  plausible route that wastes distance, which is what the condition tests.
- Our path encoder is ~50 k parameters against the paper's 12,960. Our observation is
  74-D rather than their 2636-D flattened depth map, so absolute parameter counts are
  not directly comparable.
- The shortcut reward's exact functional form is not recoverable from the paper. Ours
  fires when the robot gains **more true geodesic progress than the reference-path
  arclength it consumed** — i.e. it cut a corner the path did not offer. See
  `RewardConfig` and `torch_env.step`.

---

## Why the physics runs on the CPU and the network on the GPU

The obvious plan on a 30-GPU-core M3 Max is "MJX on Metal". Measured on this machine,
that is the wrong plan:

| Physics backend | Throughput |
|---|---|
| **C MuJoCo, `mujoco.rollout`, 14 threads** | **3,480,000 steps/s** |
| C MuJoCo, single thread | 107,000 steps/s |
| MJX on `jax-mps` Metal backend (B=32768, async dispatch) | 47,800 steps/s |
| MJX on JAX CPU backend (B=2048) | 14,700 steps/s |

MJX-on-Metal plateaus from B=4096 onward — **dispatch-bound, not compute-bound**. A
MuJoCo step is hundreds of tiny branchy kernels; XLA:CUDA fuses them, while the
community `jax-mps` PJRT plugin lowers op-by-op onto MPSGraph. (Apple's official
`jax-metal` has been unmaintained since October 2024.)

The GPU itself is fine — it just wants big kernels:

| Workload | CPU | Metal |
|---|---|---|
| 4096³ matmul | 1.00 TFLOP/s | **8.24 TFLOP/s** |
| PPO epoch pass, paper-sized actor | 1908 ms | **254 ms** |

Hence the split: **closed-form dynamics + ray casting batched on the GPU**, gradients
on the GPU, and MuJoCo reserved for evaluation and visualization where its fidelity
earns its cost.

---

## Install

```bash
git clone <this repo> && cd path-conditioned-nav
uv sync
```

Requires Python ≥3.11. `uv` handles everything else (torch, numpy, scipy, mujoco).

## Usage

```bash
# Train (full run: 4096 envs, 180 arenas)
uv run scripts/train.py --num-envs 4096 --num-maps 180 --iterations 3000

# Quick local check
uv run scripts/train.py --num-envs 256 --num-maps 8 --iterations 20 --device cpu

# Evaluate across all reference-path conditions, on held-out maps
uv run scripts/evaluate.py runs/main/policy_final.pt

# ...and re-run the sweep under real MuJoCo physics (sim-to-sim)
uv run scripts/evaluate.py runs/main/policy_final.pt --mujoco

# Watch it drive
uv run scripts/visualize.py runs/main/policy_final.pt --quality SUBOPTIMAL

# Tests
uv run pytest
```

---

## The experiment

`scripts/evaluate.py` sweeps the reference-path condition and reports success rate and
path efficiency for each:

| Condition | Reference path the policy is handed |
|---|---|
| `OPTIMAL` | A\* route on the PRM |
| `NOISY` | optimal + smooth displacement ≤ 1 m |
| `SUBOPTIMAL` | biased-GBFS route — plausible but wasteful |
| `DETOURED` | suboptimal + displacement ≤ 2 m |
| `WRONG_GOAL` | a well-formed route to a *different* goal |
| `NONE` | no path at all |

The paper's claim predicts a specific signature: **success rate stays roughly flat
across conditions** (the policy never becomes dependent on guidance) while
**efficiency improves when guidance is good**. A policy satisfying only the first has
learned to ignore the path; only the second, to follow it blindly.

**Environment validation.** Before any learning, a scripted pure-pursuit controller
(no obstacle avoidance, ignores the scan entirely) already shows the intended ordering,
confirming the reward and path machinery are wired correctly:

| Condition | Success |
|---|---|
| `OPTIMAL` | 67.9% |
| `SUBOPTIMAL` | 47.8% |
| `WRONG_GOAL` | 21.3% |
| `NONE` | 21.4% |

## Measured training throughput

M3 Max (14 CPU / 30 GPU cores, 36 GB), 2048 envs, MPS backend:

```
rollout   1.52 s     (env.step 0.62 s, policy.act 0.83 s)
update    7.02 s
total     8.54 s/iteration  ->  5,755 env-steps/s
```

---

## Layout

```
src/pcnav/
  config.py            all tunables; one ExperimentConfig fully describes a run
  maps.py              procedural arenas, inflated occupancy, geodesic fields
  planning.py          PRM, A*, biased GBFS, path resampling and corruption
  path_library.py      precomputed route tables (cached) so resets stay on-GPU
  envs/
    torch_env.py       batched env: dynamics, ray casting, rewards, resets
    mujoco_env.py      same logic, real physics — evaluation and visualization
  models/
    path_encoder.py    waypoint self-attn -> cross-attn; temporally consistent dropout
    actor_critic.py    asymmetric actor-critic
  algorithms/
    ppo.py             clipped PPO, value clipping, KL-adaptive LR
    runner.py          rollout / update / logging / checkpoint loop
  sim/
    mjcf.py            MapData -> MJCF scene, skid-steer inverse kinematics
    viewer.py          passive viewer with path + goal markers, offscreen render
  utils/               metric tracking, logging, seeding
scripts/               train.py, evaluate.py, visualize.py
tests/                 27 tests over maps, env geometry, rewards, models
```

### Two implementation details that are easy to get wrong

1. **Attention masking.** 10% of training episodes have *no* reference path, so every
   waypoint is masked. Attention over a fully-masked sequence yields NaNs; those rows
   get a dummy attendable slot and their output is forced to exact zero. That clean
   zero is the "no guidance available" signal.

2. **Dropout must be replayable.** Any dropout that redraws its mask per forward pass
   means a PPO update re-scores actions through a *different* network than produced
   them, silently corrupting the importance ratio. All stochastic regularization lives
   in `TemporallyConsistentDropout`, whose mask is recorded in the rollout buffer and
   replayed at update time — which also matches the paper's named regularizer. There is
   deliberately no dropout inside the attention stack. (`tests/test_models.py` guards
   this.)

## Citation

```bibtex
@article{haro2026pathconditioned,
  title  = {Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation},
  author = {Haro, Mateo and Richter, Julia and Yang, Fan and Cadena, Cesar and Hutter, Marco},
  journal = {arXiv preprint arXiv:2603.13888},
  year   = {2026}
}
```
