# Path-Conditioned RL Local Planning — Wheeled Robot, Apple Silicon

A reproduction of **"Path-conditioned Reinforcement Learning-based Local Planning for
Long-Range Navigation"** (Haro, Richter, Yang, Cadena, Hutter — [arXiv:2603.13888](https://arxiv.org/abs/2603.13888)),
retargeted from an Isaac Sim / quadruped setup to a **wheeled robot trained end-to-end
on an M3 Max**.

The paper's claim in one sentence: *give a navigation policy a reference path as an
**observation** rather than something it is rewarded for following, and it learns to
exploit good paths while staying robust to misleading ones.*

---

## Result

Deterministic evaluation, held-out maps, 200 episodes per condition. Baseline is the
same architecture trained **without ever seeing a path** (0.44 uniformly).

| condition | success | efficiency |
|---|---|---|
| `OPTIMAL` | **0.99** | **1.12** |
| `NOISY` | 0.99 | 1.14 |
| `SUBOPTIMAL` | 0.98 | 1.22 |
| `DETOURED` | 0.98 | 1.38 |
| `WRONG_GOAL` | **0.24** | 1.50 |
| `NONE` | 0.43 | 1.86 |

**Reproduced:** path exploitation (0.43 → 0.99), and the paper's predicted signature —
success flat across `OPTIMAL → DETOURED` while efficiency degrades monotonically. A
tracker's success would collapse; a path-ignorer's efficiency would be flat.

**Did not reproduce:** `WRONG_GOAL` robustness. Handed a well-formed route to the wrong
place, the policy follows it (0.24, *below* the no-path baseline). Likely structural —
the policy sees 15 m of path against 60 m routes, so a wrong-goal path is locally
indistinguishable from a correct one.

**Sim-to-sim:** the policy transfers to MuJoCo rigid-body physics at 0.97 on `OPTIMAL`,
ordering preserved. Not an experiment in the paper.

---

## Why this is not a line-by-line port

The paper trains in **Isaac Sim / Isaac Lab** on an **RTX 4090** (39.5 h), with a Unitree
B2W consuming **40×64 depth images**. None of that runs on Apple Silicon: Isaac is
CUDA + Linux/Windows only, and MJX has **no batched renderer** (`madrona-mjx` is
CUDA-only). So the substrate was swapped and the contribution kept.

| | paper | this repo |
|---|---|---|
| Simulator (training) | Isaac Sim / Isaac Lab | vectorized PyTorch env |
| Simulator (eval/viz) | — | **MuJoCo** |
| Robot | Unitree B2W (wheeled quadruped) | differential-drive base |
| Action | `(v_x, v_y, ω)` @ 5 Hz | `(v_x, ω)` @ 5 Hz — cannot strafe |
| Perception | 40×64 depth, pretrained CNN | 64-ray scan, **same** 105° FOV / 10 m |
| Memory | SRU (Yang et al. 2025) | **same** (SRU-GRU variant) |
| Path generation | PRM + A\*, biased GBFS, ±1 m noise | **same**, smooth 0.4 m noise |
| Path encoder | self-attn → cross-attn, 15 waypoints | **same** |
| Reward | goal-reaching + shortcut, **no path-following term** | **same** |
| Learning | asymmetric-critic PPO (rsl-rl) | **same** |
| Base policy | Yang et al. 2025, pretrained on synthetic depth | our own path-free navigator |
| Environment | procedural mazes, 30 m train / 50 m eval | procedural mazes, 45 m |

**Guesses forced by the paper**, which does not publish them: the path-quality mixture
(30/25/20/10/5/10), the shortcut reward's functional form, and the exact SRU variant.

**Deliberate deviations, with reasons.** Perturbation is smooth 0.4 m rather than IID
1 m, because 1 m puts a path through walls in a 2.8 m corridor. Training is at 45 m
rather than 30 m, because a path-blind policy solved 30 m mazes at 0.97 — leaving
nothing to measure. **Deep Mutual Learning is not implemented.**

---

## Why physics runs on the CPU and the network on the GPU

The obvious plan on a 30-GPU-core M3 Max is "MJX on Metal". Measured, that is the wrong
plan:

| physics backend | steps/s |
|---|---|
| **C MuJoCo, `mujoco.rollout`, 14 threads** | **3,480,000** |
| C MuJoCo, single thread | 107,000 |
| MJX on `jax-mps` Metal (B=32768, async) | 47,800 |
| MJX on JAX CPU (B=2048) | 14,700 |

MJX-on-Metal plateaus from B=4096 — **dispatch-bound, not compute-bound**. A MuJoCo step
is hundreds of tiny branchy kernels; XLA:CUDA fuses them, while the community `jax-mps`
PJRT plugin lowers op-by-op onto MPSGraph. (Apple's official `jax-metal` has been
unmaintained since October 2024.)

The GPU is fine — it wants big kernels: **8.24 TFLOP/s** on a 4096³ matmul against 1.00
on CPU. Hence closed-form dynamics and ray casting batched on the GPU, gradients on the
GPU, and MuJoCo reserved for evaluation where its fidelity earns its cost.

---

## Install and use

```bash
uv sync                                    # Python >=3.11; uv handles the rest

# Two-stage training, as the paper does (it warm-starts from a pretrained navigator)
uv run scripts/train.py --num-envs 4096 --num-maps 80 --iterations 500 \
    --fixed-quality NONE --run-name base           # stage 1: path-free navigator
uv run scripts/train.py --num-envs 4096 --num-maps 80 --iterations 700 \
    --init-from runs/base/policy_final.pt --run-name mixture   # stage 2: path mixture

uv run scripts/evaluate.py runs/mixture/policy_final.pt            # ablation table
uv run scripts/evaluate.py runs/mixture/policy_final.pt --mujoco   # + sim-to-sim
uv run scripts/visualize.py runs/mixture/policy_final.pt --quality SUBOPTIMAL

uv run python tools/render_map.py     # look at an arena before trusting it
uv run pytest
```

---

## Layout

```
src/pcnav/
  config.py            all tunables; one ExperimentConfig fully describes a run
  maps.py              procedural mazes, inflated occupancy, geodesic fields
  planning.py          PRM, A*, biased GBFS, path resampling and corruption
  path_library.py      precomputed route tables (cached) so resets stay on-GPU
  envs/
    torch_env.py       batched env: dynamics, ray casting, rewards, resets
    mujoco_env.py      same logic, real physics — evaluation and visualization
  models/
    recurrent.py       Spatially-Enhanced Recurrent Unit (Yang et al. 2025)
    path_encoder.py    waypoint self-attn -> cross-attn; temporally consistent dropout
    actor_critic.py    recurrent asymmetric actor-critic
  algorithms/
    ppo.py             clipped PPO, sequence minibatching, KL-adaptive LR
    runner.py          rollout / update / logging / checkpoint loop
  sim/
    mjcf.py            MapData -> MJCF scene, differential-drive kinematics
    depth.py           analytic depth rendering, no renderer required
    viewer.py          passive viewer with path + goal markers, offscreen render
scripts/               train.py, evaluate.py, visualize.py
tools/                 diagnostics -- see tools/README.md
tests/                 68 tests over maps, geometry, rewards, models, recurrence
```

---

## Reading list

Ordered by marginal value to someone who already knows PPO and RL. Each entry connects
to something that actually went wrong here.

**1. Ng, Harada & Russell (1999), _Policy Invariance Under Reward Transformations_.**
The theory behind the worst bug in this project. The shortcut reward was farmable by
oscillation — the policy drove backwards to collect it — and the fix (a non-decreasing
running maximum) works for reasons this paper makes precise. Potential-based shaping is
*provably* policy-invariant and *provably* unfarmable. Read it and "is this shaping term
telescoping?" becomes reflex.

**2. Yang et al. (2025), _Spatially-Enhanced Recurrent Memory_ ([arXiv:2506.05997](https://arxiv.org/abs/2506.05997)).**
The SRU implemented in `models/recurrent.py`, and the base navigator this paper builds
on. Short. Their spatial-memorization benchmark design is as instructive as the
architecture.

**3. Jayakumar et al. (ICLR 2020), _Multiplicative Interactions and Where to Find Them_.**
The *why* behind the SRU, generalized: a formal account of what multiplicative
interactions buy that additive layers cannot cheaply reach. Pair with **FiLM** (Perez et
al. 2018). Together they turn "a GRU with one extra multiplicative term" from a trick
into an instance of a principle.

**4. Kapturowski et al. (ICLR 2019), _R2D2_.** The definitive treatment of hidden state
in RL — stored states, burn-in, staleness. This codebase hit the update-vs-collection
consistency hazard three times (dropout mask, KL estimator, recurrent replay); this is
where the field worked it out properly.

**5. Teacher–student privileged distillation.** The natural next project. The asymmetric
critic here uses privileged information, but **this repo does not do distillation** —
that would mean training a privileged teacher and transferring it into a sensor-limited
student. Start with _Learning by Cheating_ (Chen et al. 2019); then Lee et al. (2020) and
Miki et al. (2022), both *Science Robotics*, both from this same lab.

**6. Domain randomization** (Tobin et al. 2017; Peng et al. 2018). The sim-to-sim result
here is a weak version of this question, and the paper uses domain randomization
*instead of* a curriculum.

**Skip**, given that background: transformer/attention fundamentals (the path encoder is
a two-layer attention block) and A\* theory (nothing difficult lived there).

---

## Two implementation details that are easy to get wrong

**Update-time behaviour must match collection-time behaviour.** PPO's importance ratio
assumes stored actions are re-scored under the *same* network that produced them. Three
separate violations occurred here — an unrecorded dropout mask, dropout inside the
attention stack, and recurrent replay from the wrong hidden state. None raised an error;
each silently corrupted the gradient. `tests/test_recurrent.py` pins the last one
bit-for-bit.

**Cross-component assumptions drift silently.** The MuJoCo robot had a 0.569 m envelope
while maps, roadmap and collision all modelled a 0.350 m disc. Invisible while a control
bug prevented the robot from turning, and instantly fatal once that was fixed — a masked
bug can look like a working system. `tests/test_env.py` now pins the two together.

## Citation

```bibtex
@article{haro2026pathconditioned,
  title  = {Path-conditioned Reinforcement Learning-based Local Planning for Long-Range Navigation},
  author = {Haro, Mateo and Richter, Julia and Yang, Fan and Cadena, Cesar and Hutter, Marco},
  journal = {arXiv preprint arXiv:2603.13888},
  year   = {2026}
}
@article{yang2025spatially,
  title  = {Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation via End-to-End Reinforcement Learning},
  author = {Yang, Fan and Frivik, Per and Hoeller, David and Wang, Chen and Cadena, Cesar and Hutter, Marco},
  journal = {The International Journal of Robotics Research},
  year   = {2025}
}
```
