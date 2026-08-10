"""Plot the path-quality ablation: the paper's headline experiment."""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["OPTIMAL", "NOISY", "SUBOPTIMAL", "DETOURED", "WRONG_GOAL", "NONE"]

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--results", default="media/results.json")
ap.add_argument("--baseline-success", type=float, default=0.44,
                help="path-blind policy's success, the level to beat")
ap.add_argument("--out", default="media/results.png")
args = ap.parse_args()

data = json.load(open(args.results))["torch"]
success = [data[k]["success_rate"] for k in ORDER]
efficiency = [data[k]["path_efficiency"] for k in ORDER]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
colours = ["#2ea8ff", "#3fbfb0", "#7bc47f", "#ffb020", "#ff4d4d", "#9aa0a6"]

ax = axes[0]
ax.bar(ORDER, success, color=colours)
ax.axhline(args.baseline_success, ls="--", c="k", lw=1.4,
           label=f"path-blind baseline ({args.baseline_success:.2f})")
ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05)
ax.set_title("Success vs reference-path quality\nflat across degradation = robustness")
ax.legend(fontsize=8)

ax = axes[1]
ax.bar(ORDER, efficiency, color=colours)
ax.axhline(1.0, ls=":", c="k", lw=1.2, label="perfect (1.0)")
ax.set_ylabel("distance travelled / optimal route")
ax.set_title("Efficiency vs reference-path quality\nlower is better; ordering = the path is being used")
ax.legend(fontsize=8)

for ax in axes:
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(args.out, dpi=120, facecolor="white")
print(f"wrote {args.out}")
