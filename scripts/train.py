#!/usr/bin/env python
"""Train the path-conditioned navigation policy.

Examples
--------
Full run on the GPU:
    uv run scripts/train.py --num-envs 4096 --iterations 3000

Quick local check:
    uv run scripts/train.py --num-envs 256 --num-maps 8 --iterations 20 --device cpu
"""

from __future__ import annotations

import argparse

import torch

from pcnav.algorithms import Runner
from pcnav.config import EnvConfig, ExperimentConfig, MapConfig, PPOConfig, TrainConfig
from pcnav.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-maps", type=int, default=180, help="procedural arenas in the pool")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--device", default=None, help="cpu | mps | cuda (auto-detected if unset)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="pcnav")
    parser.add_argument("--run-dir", default="runs")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=200)
    parser.add_argument(
        "--fixed-quality", default=None,
        help="pin one reference-path condition (e.g. NONE) instead of sampling the mixture",
    )
    parser.add_argument(
        "--init-from", default=None,
        help="warm-start from a checkpoint, as the paper does from a pretrained base",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    config = ExperimentConfig(
        env=EnvConfig(
            num_envs=args.num_envs,
            device=device,
            seed=args.seed,
            maps=MapConfig(num_maps=args.num_maps),
            fixed_path_quality=args.fixed_quality,
        ),
        ppo=PPOConfig(rollout_steps=args.rollout_steps),
        train=TrainConfig(
            total_iterations=args.iterations,
            run_dir=args.run_dir,
            run_name=args.run_name,
            log_interval=args.log_interval,
            checkpoint_interval=args.checkpoint_interval,
            init_from=args.init_from,
        ),
    )

    print(f"device={device}  envs={args.num_envs}  maps={args.num_maps}")
    print("building map pool and roadmaps (one-time CPU cost)...", flush=True)
    runner = Runner(config)
    print(f"run directory: {runner.run_dir}", flush=True)
    runner.train()


if __name__ == "__main__":
    main()
