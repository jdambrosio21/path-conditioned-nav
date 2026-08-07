"""Top-down render of a generated arena: maze, geodesic field, and reference paths.

Looking at the environment is the cheapest sanity check available, and the one
most often skipped. Two versions of this benchmark shipped before anyone plotted
one -- both were solvable without a reference path at all.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

from pcnav.maps import generate_map, world_to_cell
from pcnav.planning import PathQuality, build_prm, make_reference_path


def draw_geometry(ax, m):
    ax.add_patch(Rectangle((0, 0), m.size, m.size, fc="none", ec="0.3", lw=2))
    for cx, cy, half_len, half_thick, yaw in m.walls:
        ax.add_patch(Rectangle(
            (-half_len, -half_thick), 2 * half_len, 2 * half_thick,
            transform=(matplotlib.transforms.Affine2D()
                       .rotate(yaw).translate(cx, cy) + ax.transData),
            fc="0.35", ec="none"))
    for cx, cy, r in m.obstacles:
        ax.add_patch(Circle((cx, cy), r, fc="0.55", ec="none"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="media/map.png")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    m = generate_map(rng)
    prm = build_prm(m, rng)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

    # --- 1. geometry ---
    ax = axes[0]
    draw_geometry(ax, m)
    ax.set_title(f"maze geometry\n{len(m.walls)} wall segments, {len(m.obstacles)} obstacles")

    # --- 2. geodesic distance field to one goal ---
    ax = axes[1]
    goal_index = 0
    field = np.where(np.isfinite(m.dist[goal_index]), m.dist[goal_index], np.nan)
    ax.imshow(field, origin="lower", extent=[0, m.size, 0, m.size], cmap="viridis")
    draw_geometry(ax, m)
    ax.plot(*m.goals[goal_index], "*", ms=22, color="#22dd55", mec="k")
    ax.set_title("true geodesic distance to goal\n(privileged: critic + shortcut reward only)")

    # --- 3. reference paths of differing quality ---
    ax = axes[2]
    draw_geometry(ax, m)
    valid = np.flatnonzero(m.starts_valid[goal_index])
    start = m.starts[valid[len(valid) // 2]]
    goal = m.goals[goal_index]
    for quality, colour, label in (
        (PathQuality.OPTIMAL, "#2ea8ff", "A* optimal"),
        (PathQuality.SUBOPTIMAL, "#ffb020", "biased GBFS"),
        (PathQuality.WRONG_GOAL, "#ff4d4d", "wrong goal"),
    ):
        path, n = make_reference_path(m, prm, start, goal, quality, rng)
        if n > 1:
            ax.plot(path[:n, 0], path[:n, 1], lw=2.5, color=colour, label=label, alpha=0.9)
    ax.plot(*start, "o", ms=11, color="w", mec="k", label="start")
    ax.plot(*goal, "*", ms=22, color="#22dd55", mec="k", label="goal")

    sc = world_to_cell(start)
    detour = m.dist[goal_index, sc[0], sc[1]] / np.hypot(*(start - goal))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"reference paths\ndetour ratio {detour:.1f}x straight line")

    for ax in axes:
        ax.set_xlim(-1, m.size + 1)
        ax.set_ylim(-1, m.size + 1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(args.out, dpi=110, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
