"""Reference path generation.

The paper builds a Probabilistic Roadmap and draws two grades of reference path
from it: A* for optimal paths, and Greedy Best-First Search under a biased
heuristic for "controlled sub-optimality". Waypoints are then perturbed by up to
1 m. We reproduce that, and add the degraded conditions the robustness ablation
needs (detoured, wrong-goal, absent).

Paths leave here already resampled to a fixed arclength spacing and zero-padded
to a fixed length, so that projecting the robot onto its path at rollout time is
a nearest-vertex lookup -- a single batched argmin on the GPU rather than a
per-environment geometric solve.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.spatial import cKDTree

from .config import (
    MAX_PATH_VERTICES,
    NUM_WAYPOINTS,
    PATH_VERTEX_SPACING_M,
    WAYPOINT_SPACING_M,
)
from .maps import RES, MapData, world_to_cell

PATH_SPACING = PATH_VERTEX_SPACING_M
PATH_MAX = MAX_PATH_VERTICES
N_WAYPOINTS = NUM_WAYPOINTS
WAYPOINT_SPACING = WAYPOINT_SPACING_M


class PathQuality(IntEnum):
    """Reference path conditions. Training mixes these; the ablation sweeps them."""

    OPTIMAL = 0     # A* on the PRM
    NOISY = 1       # A* with up to 1 m per-waypoint perturbation
    SUBOPTIMAL = 2  # biased GBFS -- plausible but not shortest
    DETOURED = 3    # forced through a random far-away via point
    WRONG_GOAL = 4  # a valid path, but to the wrong place
    NONE = 5        # no path at all; policy must fall back on goal-reaching


@dataclass
class PRM:
    nodes: np.ndarray          # (N, 2) world-frame node positions
    neighbors: list[np.ndarray]  # per-node neighbour indices
    costs: list[np.ndarray]      # per-node edge costs, aligned with `neighbors`
    tree: cKDTree


def _segment_free(a: np.ndarray, b: np.ndarray, free: np.ndarray, res: float = RES) -> bool:
    """True if the straight segment a->b stays inside inflated free space."""
    d = float(np.linalg.norm(b - a))
    n = max(2, int(np.ceil(d / (res * 0.5))))
    pts = a[None] + (b - a)[None] * np.linspace(0.0, 1.0, n)[:, None]
    rc = world_to_cell(pts, res)
    h, w = free.shape
    if np.any((rc[:, 0] < 0) | (rc[:, 0] >= h) | (rc[:, 1] < 0) | (rc[:, 1] >= w)):
        return False
    return bool(free[rc[:, 0], rc[:, 1]].all())


def build_prm(md: MapData, rng: np.random.Generator, n_nodes: int = 400, k: int = 10) -> PRM:
    """Sample a roadmap over free space and connect each node to its k nearest."""
    free_rc = np.argwhere(md.free)
    pick = rng.choice(len(free_rc), size=min(n_nodes, len(free_rc)), replace=False)
    nodes = (free_rc[pick][:, ::-1].astype(np.float64) + 0.5) * md.res

    tree = cKDTree(nodes)
    _, idx = tree.query(nodes, k=min(k + 1, len(nodes)))

    neighbors, costs = [], []
    for i in range(len(nodes)):
        nb, cs = [], []
        for j in idx[i][1:]:
            if _segment_free(nodes[i], nodes[j], md.free, md.res):
                nb.append(j)
                cs.append(float(np.linalg.norm(nodes[i] - nodes[j])))
        neighbors.append(np.array(nb, dtype=np.int64))
        costs.append(np.array(cs, dtype=np.float64))
    return PRM(nodes=nodes, neighbors=neighbors, costs=costs, tree=tree)


def _reconstruct(came_from: dict[int, int], node: int) -> list[int]:
    path = [node]
    while node in came_from:
        node = came_from[node]
        path.append(node)
    return path[::-1]


def astar(prm: PRM, start: int, goal: int) -> list[int] | None:
    """Shortest roadmap path. Euclidean heuristic is admissible here, so this is optimal."""
    g = {start: 0.0}
    came_from: dict[int, int] = {}
    goal_xy = prm.nodes[goal]
    open_heap = [(float(np.linalg.norm(prm.nodes[start] - goal_xy)), start)]
    closed: set[int] = set()

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            return _reconstruct(came_from, cur)
        if cur in closed:
            continue
        closed.add(cur)
        for nb, c in zip(prm.neighbors[cur], prm.costs[cur], strict=True):
            ng = g[cur] + c
            if ng < g.get(nb, np.inf):
                g[nb] = ng
                came_from[nb] = cur
                f = ng + float(np.linalg.norm(prm.nodes[nb] - goal_xy))
                heapq.heappush(open_heap, (f, int(nb)))
    return None


def gbfs_biased(
    prm: PRM,
    start: int,
    goal: int,
    rng: np.random.Generator,
    bias_strength: float = 0.45,
) -> list[int] | None:
    """Greedy best-first search under a spatially smooth, randomly biased heuristic.

    GBFS ignores accumulated cost, so it is already suboptimal. Multiplying the
    heuristic by a smooth random field makes that suboptimality *structured* --
    the path commits to a plausible-looking but needlessly long corridor, which is
    exactly the failure mode a real upstream global planner exhibits. Pure random
    noise would instead produce jagged paths no planner would ever emit.
    """
    # Two sinusoidal lobes give a smooth multiplicative field in [1-b, 1+b].
    freq = rng.uniform(0.05, 0.18, size=2)
    phase = rng.uniform(0, 2 * np.pi, size=2)
    amp = rng.uniform(0.5, 1.0, size=2)
    amp = amp / amp.sum()

    def h(i: int) -> float:
        p = prm.nodes[i]
        f = (amp * np.sin(freq * p[:2] * 2 * np.pi + phase)).sum()
        return float(np.linalg.norm(p - prm.nodes[goal])) * (1.0 + bias_strength * f)

    came_from: dict[int, int] = {}
    open_heap = [(h(start), start)]
    seen = {start}

    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == goal:
            return _reconstruct(came_from, cur)
        for nb in prm.neighbors[cur]:
            nb = int(nb)
            if nb not in seen:
                seen.add(nb)
                came_from[nb] = cur
                heapq.heappush(open_heap, (h(nb), nb))
    return None


def resample(polyline: np.ndarray, spacing: float = PATH_SPACING) -> np.ndarray:
    """Resample a polyline to uniform arclength spacing."""
    if len(polyline) < 2:
        return polyline.astype(np.float32)
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return polyline[:1].astype(np.float32)
    n = max(2, int(np.ceil(total / spacing)) + 1)
    q = np.linspace(0.0, total, n)
    out = np.stack([np.interp(q, s, polyline[:, d]) for d in range(2)], axis=1)
    return out.astype(np.float32)


def perturb(polyline: np.ndarray, rng: np.random.Generator, max_offset: float = 1.0) -> np.ndarray:
    """Displace the path by a smooth random offset of up to `max_offset` metres.

    The paper perturbs waypoints by up to 1 m. Applying a smoothly varying offset
    rather than per-vertex IID noise keeps the result a path rather than a
    zigzag, so the policy learns to tolerate *registration* error -- the realistic
    defect when a global plan is built on a stale or misaligned map.
    """
    n = len(polyline)
    if n < 2:
        return polyline
    n_ctrl = max(2, n // 16)
    ctrl = rng.uniform(-1.0, 1.0, size=(n_ctrl, 2))
    t = np.linspace(0, 1, n)
    tc = np.linspace(0, 1, n_ctrl)
    off = np.stack([np.interp(t, tc, ctrl[:, d]) for d in range(2)], axis=1)
    mag = np.linalg.norm(off, axis=1, keepdims=True).clip(1e-6)
    off = off / mag * np.minimum(mag, 1.0) * max_offset
    return (polyline + off).astype(np.float32)


def make_reference_path(
    md: MapData,
    prm: PRM,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    quality: PathQuality,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Build one reference path, padded to (PATH_MAX, 2). Returns (path, n_valid).

    n_valid == 0 encodes "no usable path", which is what the policy sees under
    PathQuality.NONE and whenever planning legitimately fails.
    """
    empty = np.zeros((PATH_MAX, 2), dtype=np.float32)
    if quality == PathQuality.NONE:
        return empty, 0

    s = int(prm.tree.query(start_xy)[1])
    if quality == PathQuality.WRONG_GOAL:
        # A perfectly well-formed path -- to somewhere else entirely.
        alt = md.goals[rng.integers(len(md.goals))]
        g = int(prm.tree.query(alt)[1])
    else:
        g = int(prm.tree.query(goal_xy)[1])

    if quality == PathQuality.SUBOPTIMAL:
        nodes = gbfs_biased(prm, s, g, rng)
    elif quality == PathQuality.DETOURED:
        via = int(rng.integers(len(prm.nodes)))
        a = astar(prm, s, via)
        b = astar(prm, via, g)
        nodes = (a[:-1] + b) if (a and b) else astar(prm, s, g)
    else:
        nodes = astar(prm, s, g)

    if not nodes or len(nodes) < 2:
        return empty, 0

    poly = prm.nodes[np.asarray(nodes)]
    # Anchor the path at the true start and (except when deliberately wrong) the goal.
    poly = np.concatenate([start_xy[None], poly], axis=0)
    if quality != PathQuality.WRONG_GOAL:
        poly = np.concatenate([poly, goal_xy[None]], axis=0)

    poly = resample(poly)
    if quality in (PathQuality.NOISY, PathQuality.SUBOPTIMAL, PathQuality.DETOURED):
        poly = perturb(poly, rng, max_offset=1.0)

    n = min(len(poly), PATH_MAX)
    out = empty.copy()
    out[:n] = poly[:n]
    if n < PATH_MAX:  # pad by repeating the tail so nearest-vertex lookups stay sane
        out[n:] = poly[n - 1]
    return out, n
