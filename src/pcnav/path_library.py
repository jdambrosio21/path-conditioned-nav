"""Precomputed reference-path library.

Planning a fresh A* path on every episode reset makes graph search, not physics
or gradients, the training bottleneck: early in training almost every environment
terminates on collision within a few steps, so thousands of searches run per
iteration on the CPU while the GPU idles.

Instead, every (map, goal, start) triple gets its optimal and suboptimal routes
planned **once** at startup and cached to disk. At reset time the environment
gathers from these tensors on-device, and the corrupted variants the ablations
need are synthesized on the GPU from the two stored routes:

    OPTIMAL     stored A* route
    NOISY       optimal + smooth random offset (<= 1 m)
    SUBOPTIMAL  stored biased-GBFS route
    DETOURED    suboptimal + larger smooth offset (<= 2 m)
    WRONG_GOAL  the optimal route to a *different* goal on the same map
    NONE        zeros, flagged invalid

The only fidelity cost is that DETOURED is now "a bad path made worse" rather
than "a path forced through a random via point". Both produce a plausible route
that wastes distance, which is what the condition is there to test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MAX_PATH_VERTICES, MapConfig
from .maps import MapData
from .planning import PathQuality, build_prm, make_reference_path

CACHE_VERSION = 1


@dataclass
class PathLibrary:
    """Dense per-(map, goal, start) route tables.

    Shapes are (num_maps, goals_per_map, starts_per_goal, MAX_PATH_VERTICES, 2)
    for the routes and (num_maps, goals_per_map, starts_per_goal) for the lengths.
    """

    optimal: np.ndarray
    optimal_len: np.ndarray
    suboptimal: np.ndarray
    suboptimal_len: np.ndarray

    @property
    def num_maps(self) -> int:
        return self.optimal.shape[0]

    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.optimal, self.optimal_len,
                                      self.suboptimal, self.suboptimal_len))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            version=CACHE_VERSION,
            optimal=self.optimal,
            optimal_len=self.optimal_len,
            suboptimal=self.suboptimal,
            suboptimal_len=self.suboptimal_len,
        )

    @classmethod
    def load(cls, path: Path) -> PathLibrary | None:
        if not path.exists():
            return None
        try:
            blob = np.load(path)
            if int(blob["version"]) != CACHE_VERSION:
                return None
            return cls(
                optimal=blob["optimal"],
                optimal_len=blob["optimal_len"],
                suboptimal=blob["suboptimal"],
                suboptimal_len=blob["suboptimal_len"],
            )
        except (OSError, KeyError, ValueError):
            # A corrupt or partially written cache should never be fatal; the
            # caller simply replans.
            return None


def cache_key(map_config: MapConfig, seed: int) -> str:
    """Stable hash over everything that changes the generated routes."""
    payload = json.dumps(
        {
            "seed": seed,
            "version": CACHE_VERSION,
            "max_vertices": MAX_PATH_VERTICES,
            **{k: v for k, v in map_config.__dict__.items()},
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def build_path_library(
    maps: list[MapData],
    map_config: MapConfig,
    rng: np.random.Generator,
    progress: bool = True,
) -> PathLibrary:
    """Plan optimal and suboptimal routes for every (map, goal, start) triple."""
    num_maps = len(maps)
    num_goals = maps[0].goals.shape[0]
    num_starts = maps[0].starts.shape[0]
    shape = (num_maps, num_goals, num_starts, MAX_PATH_VERTICES, 2)

    optimal = np.zeros(shape, dtype=np.float32)
    optimal_len = np.zeros(shape[:3], dtype=np.int32)
    suboptimal = np.zeros(shape, dtype=np.float32)
    suboptimal_len = np.zeros(shape[:3], dtype=np.int32)

    total = num_maps * num_goals * num_starts
    if progress:
        print(f"planning {total:,} route pairs across {num_maps} maps...", flush=True)

    for map_idx, map_data in enumerate(maps):
        roadmap = build_prm(
            map_data, rng, n_nodes=map_config.roadmap_nodes, k=map_config.roadmap_neighbors
        )
        for goal_idx in range(num_goals):
            goal_xy = map_data.goals[goal_idx]
            for start_idx in range(num_starts):
                start_xy = map_data.starts[start_idx]
                optimal[map_idx, goal_idx, start_idx], optimal_len[map_idx, goal_idx, start_idx] = (
                    make_reference_path(
                        map_data, roadmap, start_xy, goal_xy, PathQuality.OPTIMAL, rng,
                        map_config.path_perturbation_m,
                    )
                )
                (
                    suboptimal[map_idx, goal_idx, start_idx],
                    suboptimal_len[map_idx, goal_idx, start_idx],
                ) = make_reference_path(
                    map_data, roadmap, start_xy, goal_xy, PathQuality.SUBOPTIMAL, rng,
                    map_config.path_perturbation_m,
                )
        if progress and (map_idx + 1) % 20 == 0:
            print(f"  {map_idx + 1}/{num_maps} maps", flush=True)

    # A start with no reachable route is unusable; fall back to the optimal route
    # so the suboptimal condition never silently degenerates into "no path".
    missing = (suboptimal_len == 0) & (optimal_len > 0)
    if missing.any():
        suboptimal[missing] = optimal[missing]
        suboptimal_len[missing] = optimal_len[missing]

    return PathLibrary(optimal, optimal_len, suboptimal, suboptimal_len)


def load_or_build(
    maps: list[MapData],
    map_config: MapConfig,
    seed: int,
    rng: np.random.Generator,
    cache_dir: Path = Path(".cache/paths"),
) -> PathLibrary:
    """Return a cached library if one matches this configuration, else build it."""
    path = Path(cache_dir) / f"paths_{cache_key(map_config, seed)}.npz"
    library = PathLibrary.load(path)
    if library is not None and library.num_maps == len(maps):
        print(f"loaded cached path library: {path}", flush=True)
        return library

    library = build_path_library(maps, map_config, rng)
    library.save(path)
    print(f"cached path library -> {path} ({library.nbytes() / 1e6:.0f} MB in memory)", flush=True)
    return library
