#!/usr/bin/env python
"""Watch a trained policy drive in MuJoCo.

The reference path is drawn as a ribbon of blue markers and the goal as a green
sphere, so you can see directly whether the policy is following guidance,
shortcutting past it, or ignoring a path that leads somewhere wrong.

Examples
--------
    uv run scripts/visualize.py runs/main/policy_final.pt
    uv run scripts/visualize.py runs/main/policy_final.pt --quality WRONG_GOAL
    uv run scripts/visualize.py runs/main/policy_final.pt --record episode.mp4
"""

from __future__ import annotations

import argparse

# Reuse the checkpoint loader so the policy is always rebuilt from its own config.
import sys
from dataclasses import replace
from pathlib import Path

from pcnav.envs.mujoco_env import MuJoCoNavEnv
from pcnav.planning import PathQuality
from pcnav.utils import seed_everything

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_policy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--quality",
        default="SUBOPTIMAL",
        choices=[q.name for q in PathQuality],
        help="reference-path condition to visualize",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-maps", type=int, default=6)
    parser.add_argument("--record", type=Path, default=None, help="save an mp4 instead of viewing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    policy, config = load_policy(args.checkpoint, num_envs=1, device="cpu")

    env_config = replace(
        config.env,
        num_envs=1,
        device="cpu",
        seed=args.seed,
        fixed_path_quality=args.quality,
    )
    env_config.maps = replace(config.env.maps, num_maps=args.num_maps)
    env = MuJoCoNavEnv(env_config)

    from pcnav.sim.viewer import PolicyViewer, render_episode_frames

    if args.record:
        frames = render_episode_frames(env, policy)
        _write_video(frames, args.record)
        print(f"wrote {args.record} ({len(frames)} frames)")
        return

    print(f"visualizing condition: {args.quality}  (close the window to stop)")
    for _ in range(args.episodes):
        PolicyViewer(env, policy).run(max_episodes=1)


def _write_video(frames, path: Path, fps: int = 5) -> None:
    """Write frames to mp4, falling back to a PNG sequence if no encoder exists."""
    try:
        import imageio.v3 as iio

        iio.imwrite(path, frames, fps=fps)
    except ImportError:
        out_dir = path.with_suffix("")
        out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        for i, frame in enumerate(frames):
            Image.fromarray(frame).save(out_dir / f"frame_{i:04d}.png")
        print(f"imageio not installed; wrote a PNG sequence to {out_dir} instead")


if __name__ == "__main__":
    main()
