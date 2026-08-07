#!/usr/bin/env python
"""Evaluate a trained policy across reference-path conditions.

This is the paper's headline experiment: sweep the quality of the reference path
and check that (a) success rate stays roughly flat even as guidance degrades, and
(b) *efficiency* improves when the guidance is good. A policy that only satisfies
(a) has learned to ignore the path; one that only satisfies (b) has learned to
follow it blindly. The claim is that path-conditioning without a path-following
reward gets you both.

Optionally re-runs the whole sweep under MuJoCo physics for a sim-to-sim check.

Examples
--------
    uv run scripts/evaluate.py runs/pcnav/policy_final.pt
    uv run scripts/evaluate.py runs/pcnav/policy_final.pt --mujoco --episodes 60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pcnav.config import ExperimentConfig
from pcnav.envs.torch_env import PathConditionedNavEnv
from pcnav.models import PathConditionedActorCritic
from pcnav.planning import PathQuality
from pcnav.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=200, help="episodes per condition")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-maps", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1234, help="held-out map seed")
    parser.add_argument("--mujoco", action="store_true", help="also evaluate under MuJoCo physics")
    parser.add_argument("--mujoco-envs", type=int, default=8)
    parser.add_argument("--mujoco-episodes", type=int, default=24)
    parser.add_argument("--output", type=Path, default=None, help="write results as JSON")
    return parser.parse_args()


def load_policy(checkpoint_path: Path, num_envs: int, device: str):
    """Restore a policy and the config it was trained under."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    policy = PathConditionedActorCritic(config.policy, num_envs).to(device)
    policy.load_state_dict(checkpoint["policy"])
    policy.eval()
    # Dropout is a training-time regularizer only; disable it for evaluation.
    policy.obs_dropout.drop_prob = 0.0
    return policy, config


@torch.no_grad()
def evaluate_condition(env, policy, target_episodes: int) -> dict[str, float]:
    """Run until `target_episodes` have finished; report aggregate outcomes."""
    observation = env.observe()
    successes = collisions = timeouts = tipovers = 0
    completed = 0
    success_steps: list[float] = []
    success_geodesic: list[float] = []

    # Geodesic distance at spawn is the length of the optimal route, so
    # (steps taken) / (optimal length) is a clean path-efficiency ratio.
    initial_geodesic = env.previous_geodesic.clone()
    step_counter = torch.zeros(env.num_envs, device=env.device)

    while completed < target_episodes:
        action = policy.act_deterministic(observation)
        observation, _, done, info = env.step(action)
        step_counter += 1

        if bool(done.any()):
            idx = done.nonzero(as_tuple=True)[0]
            for i in idx.tolist():
                completed += 1
                if bool(info["success"][i]):
                    successes += 1
                    success_steps.append(float(step_counter[i]))
                    success_geodesic.append(float(initial_geodesic[i]))
                elif bool(info.get("tipped_over", torch.zeros_like(done))[i]):
                    tipovers += 1
                elif bool(info["collision"][i]):
                    collisions += 1
                else:
                    timeouts += 1
            step_counter[idx] = 0.0
            initial_geodesic[idx] = env.previous_geodesic[idx]

    total = max(completed, 1)
    # Distance travelled per metre of optimal route: 1.0 is perfect efficiency.
    from pcnav.config import MAX_FORWARD_SPEED, NAV_POLICY_HZ

    efficiency = float("nan")
    if success_steps:
        travelled = [
            steps / NAV_POLICY_HZ * MAX_FORWARD_SPEED for steps in success_steps
        ]
        ratios = [t / max(g, 1e-3) for t, g in zip(travelled, success_geodesic, strict=True)]
        efficiency = sum(ratios) / len(ratios)

    return {
        "episodes": completed,
        "success_rate": successes / total,
        "collision_rate": collisions / total,
        "tipover_rate": tipovers / total,
        "timeout_rate": timeouts / total,
        "mean_steps_to_goal": (
            sum(success_steps) / len(success_steps) if success_steps else float("nan")
        ),
        "path_efficiency": efficiency,
    }


def sweep(env_factory, policy, episodes: int, label: str) -> dict[str, dict]:
    """Evaluate every path-quality condition with one env per condition."""
    results: dict[str, dict] = {}
    print(f"\n=== {label} ===")
    header = (
        f"{'condition':<12} {'succ':>6} {'coll':>6} {'tip':>6} {'t/o':>6} "
        f"{'steps':>7} {'effic':>7}"
    )
    print(header)
    print("-" * len(header))

    for quality in PathQuality:
        env = env_factory(quality.name)
        stats = evaluate_condition(env, policy, episodes)
        results[quality.name] = stats
        print(
            f"{quality.name:<12} {stats['success_rate']:>6.2f} {stats['collision_rate']:>6.2f} "
            f"{stats['tipover_rate']:>6.2f} {stats['timeout_rate']:>6.2f} "
            f"{stats['mean_steps_to_goal']:>7.1f} {stats['path_efficiency']:>7.2f}"
        )
    return results


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    policy, config = load_policy(args.checkpoint, args.num_envs, args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"loaded {args.checkpoint} (trained {checkpoint['env_steps']:,} env steps)")

    # Held-out maps: a different seed from training, so this measures generalization.
    from dataclasses import replace

    def torch_factory(quality_name: str):
        env_config = replace(
            config.env,
            num_envs=args.num_envs,
            device=args.device,
            seed=args.seed,
            fixed_path_quality=quality_name,
        )
        env_config.maps = replace(config.env.maps, num_maps=args.num_maps)
        return PathConditionedNavEnv(env_config)

    all_results = {
        "torch": sweep(torch_factory, policy, args.episodes, "idealized dynamics (torch)")
    }

    if args.mujoco:
        from pcnav.envs.mujoco_env import MuJoCoNavEnv

        mujoco_policy, _ = load_policy(args.checkpoint, args.mujoco_envs, "cpu")

        def mujoco_factory(quality_name: str):
            env_config = replace(
                config.env,
                num_envs=args.mujoco_envs,
                device="cpu",
                seed=args.seed,
                fixed_path_quality=quality_name,
            )
            env_config.maps = replace(config.env.maps, num_maps=min(args.num_maps, 12))
            return MuJoCoNavEnv(env_config)

        all_results["mujoco"] = sweep(
            mujoco_factory, mujoco_policy, args.mujoco_episodes, "MuJoCo physics (sim-to-sim)"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2))
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
