"""Procedural map generation.

Each map is a 30x30 m arena (the paper's training size) populated with circular
obstacles. Circles are used rather than meshes because ray-circle intersection is
closed-form, which lets the whole perception step run as a batched GPU tensor op.

For every map we precompute, on the CPU and once:
  * a robot-radius-inflated occupancy grid,
  * a set of candidate goals drawn from the largest free connected component,
  * a true geodesic distance-to-goal field per candidate goal.

The distance fields are privileged information. They never enter the actor's
observation -- they feed the asymmetric critic and the shortcut reward only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

from .config import ARENA_SIZE_M, GRID_RESOLUTION_M, ROBOT_RADIUS_M

# Short local aliases keep the geometry code below readable.
ARENA = ARENA_SIZE_M
RES = GRID_RESOLUTION_M
ROBOT_RADIUS = ROBOT_RADIUS_M


@dataclass
class MapData:
    """One procedurally generated arena plus its precomputed planning structures."""

    obstacles: np.ndarray     # (K_obs, 3) float32 -- cx, cy, r (true radii, uninflated)
    free: np.ndarray          # (H, W) bool  -- inflated free space
    goals: np.ndarray         # (G, 2) float32 -- candidate goal positions, world frame
    dist: np.ndarray          # (G, H, W) float32 -- geodesic distance to each goal (m)
    starts: np.ndarray        # (S, 2) float32 -- candidate starts, shared across goals
    starts_valid: np.ndarray  # (G, S) bool -- start is reachable and far enough from goal
    size: float = ARENA
    res: float = RES

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape


def world_to_cell(xy: np.ndarray, res: float = RES) -> np.ndarray:
    """World metres -> (row, col) integer grid cell."""
    c = np.floor(np.asarray(xy) / res).astype(np.int64)
    return c[..., ::-1]  # (x, y) -> (row=y, col=x)


def cell_to_world(rc: np.ndarray, res: float = RES) -> np.ndarray:
    """(row, col) grid cell -> world metres at the cell centre."""
    rc = np.asarray(rc)
    xy = (rc[..., ::-1].astype(np.float64) + 0.5) * res
    return xy


def _inflated_free(obstacles: np.ndarray, size: float, res: float) -> np.ndarray:
    """Occupancy grid with obstacles dilated by the robot radius.

    Working in inflated space lets us treat the robot as a point for planning and
    for the geodesic field, which is what makes the distance fields meaningful as
    a "how far must I actually drive" signal.
    """
    n = int(round(size / res))
    ys, xs = np.mgrid[0:n, 0:n]
    cx = (xs + 0.5) * res
    cy = (ys + 0.5) * res

    free = np.ones((n, n), dtype=bool)
    for ox, oy, r in obstacles:
        free &= (cx - ox) ** 2 + (cy - oy) ** 2 > (r + ROBOT_RADIUS) ** 2

    # Arena walls: anything within a robot radius of the boundary is unreachable.
    m = int(np.ceil(ROBOT_RADIUS / res))
    free[:m, :] = free[-m:, :] = free[:, :m] = free[:, -m:] = False
    return free


def _grid_graph(free: np.ndarray, res: float) -> tuple[csr_matrix, np.ndarray]:
    """8-connected sparse graph over free cells, plus the cell->node index map."""
    h, w = free.shape
    idx = np.full((h, w), -1, dtype=np.int64)
    n_free = int(free.sum())
    idx[free] = np.arange(n_free)

    rows, cols, data = [], [], []
    # Only four offsets are needed; each edge is added symmetrically below.
    diagonal = res * np.sqrt(2)
    for dr, dc, cost in ((0, 1, res), (1, 0, res), (1, 1, diagonal), (1, -1, diagonal)):
        source = idx[max(0, -dr):h - max(0, dr), max(0, -dc):w - max(0, dc)]
        target = idx[max(0, dr):h + min(0, dr), max(0, dc):w + min(0, dc)]
        both_free = (source >= 0) & (target >= 0)
        source_nodes, target_nodes = source[both_free], target[both_free]
        edge_costs = np.full(source_nodes.size, cost)
        # Undirected: add both orientations of every edge.
        rows.extend([source_nodes, target_nodes])
        cols.extend([target_nodes, source_nodes])
        data.extend([edge_costs, edge_costs])

    graph = csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_free, n_free),
    )
    return graph, idx


def _largest_component(graph: csr_matrix, n_free: int) -> np.ndarray:
    """Boolean mask over free-cell nodes selecting the largest connected component.

    Procedural obstacle fields routinely seal off pockets. Sampling starts or goals
    inside a sealed pocket would produce unsolvable episodes that silently poison
    the success-rate metric, so we restrict everything to one component.
    """
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp == 1:
        return np.ones(n_free, dtype=bool)
    sizes = np.bincount(labels)
    return labels == sizes.argmax()


def generate_map(
    rng: np.random.Generator,
    size: float = ARENA,
    n_obstacles: tuple[int, int] = (18, 45),
    radius: tuple[float, float] = (0.30, 1.60),
    n_goals: int = 8,
    n_starts: int = 12,
    min_goal_dist: float = 12.0,
) -> MapData:
    """Generate one arena with candidate start/goal pairs and geodesic fields.

    Start/goal pairs are rejected unless separated by at least ``min_goal_dist``
    metres of *geodesic* distance, so every episode is a genuine long-range
    navigation problem rather than a short hop.
    """
    m = int(rng.integers(*n_obstacles))
    obstacles = np.stack(
        [
            rng.uniform(2.0, size - 2.0, m),
            rng.uniform(2.0, size - 2.0, m),
            rng.uniform(radius[0], radius[1], m),
        ],
        axis=1,
    ).astype(np.float32)

    free = _inflated_free(obstacles, size, RES)
    graph, idx = _grid_graph(free, RES)
    n_free = int(free.sum())
    if n_free < 500:
        return generate_map(rng, size, n_obstacles, radius, n_goals, n_starts, min_goal_dist)

    keep = _largest_component(graph, n_free)
    node_rc = np.argwhere(free)                 # (n_free, 2) row/col per node
    comp_rc = node_rc[keep]                     # cells inside the main component
    if len(comp_rc) < 400:
        return generate_map(rng, size, n_obstacles, radius, n_goals, n_starts, min_goal_dist)

    goal_nodes = rng.choice(np.flatnonzero(keep), size=n_goals, replace=False)
    dist_flat = dijkstra(graph, directed=False, indices=goal_nodes)  # (K, n_free)

    h, w = free.shape
    dist = np.full((n_goals, h, w), np.inf, dtype=np.float32)
    rr, cc = node_rc[:, 0], node_rc[:, 1]
    for k in range(n_goals):
        dist[k, rr, cc] = dist_flat[k]

    goals = cell_to_world(node_rc[goal_nodes]).astype(np.float32)

    # Candidate starts are shared across all goals rather than sampled per goal.
    # That is what lets the WRONG_GOAL condition reuse a precomputed route: the
    # path to the wrong goal genuinely begins where the robot is standing.
    start_pool = np.flatnonzero(keep)
    start_nodes = rng.choice(start_pool, size=n_starts, replace=start_pool.size < n_starts)
    starts = cell_to_world(node_rc[start_nodes]).astype(np.float32)

    # A (goal, start) pair is usable only if the start can actually reach the goal
    # and is far enough away for the episode to be a long-range problem.
    starts_valid = np.zeros((n_goals, n_starts), dtype=bool)
    for k in range(n_goals):
        d = dist_flat[k][start_nodes]
        starts_valid[k] = np.isfinite(d) & (d >= min_goal_dist)
        if not starts_valid[k].any():  # sparse map: relax rather than discard
            starts_valid[k] = np.isfinite(d) & (d >= 0.5 * min_goal_dist)
        if not starts_valid[k].any():
            starts_valid[k] = np.isfinite(d)

    if not starts_valid.any():
        return generate_map(rng, size, n_obstacles, radius, n_goals, n_starts, min_goal_dist)

    return MapData(
        obstacles=obstacles,
        free=free,
        goals=goals,
        dist=dist,
        starts=starts,
        starts_valid=starts_valid,
    )


def generate_map_set(n_maps: int, seed: int = 0, **kw) -> list[MapData]:
    """Generate a pool of maps. The paper trains across 180 procedural arenas."""
    rng = np.random.default_rng(seed)
    return [generate_map(rng, **kw) for _ in range(n_maps)]
