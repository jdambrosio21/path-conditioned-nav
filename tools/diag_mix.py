"""Mixed-condition diagnostic: does the critic fix restore per-condition learning?"""
import torch
from pcnav.algorithms import Runner
from pcnav.config import EnvConfig, ExperimentConfig, MapConfig, PPOConfig, TrainConfig
from pcnav.utils import seed_everything

seed_everything(0)
cfg = ExperimentConfig(
    env=EnvConfig(num_envs=1024, device="mps", seed=0, maps=MapConfig(num_maps=30)),
    ppo=PPOConfig(rollout_steps=24),
    train=TrainConfig(total_iterations=120, run_dir="runs/diag", run_name="mixed",
                      log_interval=20, checkpoint_interval=10**9))
r = Runner(cfg); r.train()
print()
for q, t in r.tracker.termination_breakdown().items():
    print(f"{q:11s} success={t['success']:.2f} collision={t['collision']:.2f} timeout={t['timeout']:.2f}")
