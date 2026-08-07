"""Isolate whether path conditioning itself is the problem.

Trains short runs pinned to a single reference-path condition. If OPTIMAL alone
learns, the fault is in the mixture or the encoder's interaction with it. If
OPTIMAL alone fails while NONE learns, the fault is in the path pathway.
"""
import sys

from pcnav.algorithms import Runner
from pcnav.config import EnvConfig, ExperimentConfig, MapConfig, PPOConfig, TrainConfig
from pcnav.utils import seed_everything

quality, iters = sys.argv[1], int(sys.argv[2])
seed_everything(0)
cfg = ExperimentConfig(
    env=EnvConfig(num_envs=1024, device="mps", seed=0, maps=MapConfig(num_maps=30),
                  fixed_path_quality=quality),
    ppo=PPOConfig(rollout_steps=24),
    train=TrainConfig(total_iterations=iters, run_dir="runs/diag", run_name=quality,
                      log_interval=20, checkpoint_interval=10**9))
r = Runner(cfg); r.train()
t = r.tracker.termination_breakdown()[quality]
print(f"\nRESULT {quality}: success={t['success']:.2f} collision={t['collision']:.2f} timeout={t['timeout']:.2f}")
