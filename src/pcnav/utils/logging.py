"""Lightweight metric tracking and console reporting.

Deliberately dependency-free: episode statistics are accumulated on-device and
only pulled to the host at log time, so logging never stalls the rollout.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import torch

from ..planning import PathQuality


class EpisodeTracker:
    """Rolling window of completed-episode outcomes, bucketed by path quality.

    Per-quality breakdown is the headline result of the paper: success should stay
    flat as the reference path degrades, while path *efficiency* improves when the
    path is good. A single aggregate number would hide exactly that.
    """

    def __init__(self, window: int = 2000):
        self.returns: deque[float] = deque(maxlen=window)
        self.lengths: deque[float] = deque(maxlen=window)
        self.outcomes: dict[int, deque[tuple[float, float]]] = {
            int(q): deque(maxlen=window) for q in PathQuality
        }
        # Termination reason per condition. "Did not succeed" is not a diagnosis --
        # colliding and timing out call for opposite fixes, so they are tracked apart.
        self.terminations: dict[int, deque[str]] = {
            int(q): deque(maxlen=window) for q in PathQuality
        }

    @torch.no_grad()
    def record(
        self,
        done: torch.Tensor,
        info: dict[str, torch.Tensor],
        episode_return: torch.Tensor,
        episode_length: torch.Tensor,
    ) -> None:
        if not bool(done.any()):
            return
        idx = done.nonzero(as_tuple=True)[0]
        self.returns.extend(episode_return[idx].tolist())
        self.lengths.extend(episode_length[idx].tolist())

        success = info["success"][idx].float()
        collision = info["collision"][idx].float()
        quality = info["path_quality"][idx]
        for quality_id, succeeded, collided in zip(
            quality.tolist(), success.tolist(), collision.tolist(), strict=True
        ):
            if succeeded > 0.5:
                reason = "success"
            elif collided > 0.5:
                reason = "collision"
            else:
                reason = "timeout"
            self.terminations[int(quality_id)].append(reason)
        for quality_id, succeeded, length in zip(
            quality.tolist(), success.tolist(), episode_length[idx].tolist(), strict=True
        ):
            self.outcomes[int(quality_id)].append((succeeded, length))

    @property
    def mean_return(self) -> float:
        return sum(self.returns) / len(self.returns) if self.returns else 0.0

    @property
    def mean_length(self) -> float:
        return sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def success_by_quality(self) -> dict[str, float]:
        out = {}
        for q in PathQuality:
            rows = self.outcomes[int(q)]
            out[q.name] = (
                sum(succeeded for succeeded, _ in rows) / len(rows) if rows else float("nan")
            )
        return out

    def length_by_quality(self) -> dict[str, float]:
        """Mean episode length among *successful* episodes -- the efficiency metric."""
        out = {}
        for q in PathQuality:
            rows = [length for succeeded, length in self.outcomes[int(q)] if succeeded > 0.5]
            out[q.name] = sum(rows) / len(rows) if rows else float("nan")
        return out


    def termination_breakdown(self) -> dict[str, dict[str, float]]:
        """Fraction of episodes ending each way, per path-quality condition."""
        out: dict[str, dict[str, float]] = {}
        for q in PathQuality:
            rows = self.terminations[int(q)]
            total = max(len(rows), 1)
            out[q.name] = {
                reason: sum(r == reason for r in rows) / total
                for reason in ("success", "collision", "timeout")
            }
        return out


class RunLogger:
    """Console + JSONL logging for one training run."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.start_time = time.time()

    def log(self, iteration: int, total_steps: int, metrics: dict) -> None:
        elapsed = time.time() - self.start_time
        record = {
            "iteration": iteration,
            "env_steps": total_steps,
            "elapsed_s": round(elapsed, 1),
            "steps_per_s": round(total_steps / max(elapsed, 1e-6)),
            **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()},
        }
        with self.metrics_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

        success = metrics.get("success_by_quality", {})
        headline = "  ".join(
            f"{name[:4]}={value:.2f}" for name, value in success.items() if value == value
        )
        print(
            f"[{iteration:5d}] steps={total_steps/1e6:6.2f}M "
            f"sps={record['steps_per_s']:>8,} "
            f"ret={metrics.get('mean_return', 0):7.2f} "
            f"kl={metrics.get('kl', 0):.4f} "
            f"lr={metrics.get('learning_rate', 0):.2e} | {headline}",
            flush=True,
        )
