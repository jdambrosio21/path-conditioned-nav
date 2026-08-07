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
    walls: np.ndarray         # (K_wall, 5) float32 -- cx, cy, half_len, half_thick, yaw
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


def wall_signed_distance(points: np.ndarray, walls: np.ndarray) -> np.ndarray:
    """Signed distance from each point to each oriented-box wall.

    Negative inside. Standard box SDF evaluated in the wall's local frame:
    ``d = |p_local| - half_extents``, then ``|max(d,0)| + min(max(d), 0)``.

    Args:
        points: (N, 2) world positions.
        walls:  (W, 5) as (cx, cy, half_length, half_thickness, yaw).
    Returns:
        (N, W) signed distances; +inf for padding walls (half_length <= 0).
    """
    if len(walls) == 0:
        return np.full((len(points), 0), np.inf, dtype=np.float64)

    cx, cy, half_len, half_thick, yaw = walls.T
    offset_x = points[:, 0:1] - cx
    offset_y = points[:, 1:2] - cy
    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    local_x = np.abs(offset_x * cos_y - offset_y * sin_y) - half_len
    local_y = np.abs(offset_x * sin_y + offset_y * cos_y) - half_thick

    outside = np.hypot(np.maximum(local_x, 0.0), np.maximum(local_y, 0.0))
    inside = np.minimum(np.maximum(local_x, local_y), 0.0)
    distance = outside + inside
    return np.where(half_len > 0, distance, np.inf)


def _wall(cx: float, cy: float, half_len: float, yaw: float, half_thick: float = 0.2):
    return [cx, cy, half_len, half_thick, yaw]


def _u_trap(rng: np.random.Generator, size: float) -> list[list[float]]:
    """Three walls forming a pocket -- the canonical local-minimum trap.

    A greedy policy heading toward a goal beyond the pocket drives straight in and
    must reverse out. This is precisely the situation a reference path resolves and
    local sensing cannot, so it is what makes path conditioning worth anything.
    """
    width = rng.uniform(3.0, 6.0)
    depth = rng.uniform(3.0, 5.0)
    yaw = rng.uniform(0, 2 * np.pi)
    cx = rng.uniform(6.0, size - 6.0)
    cy = rng.uniform(6.0, size - 6.0)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    def place(local_x, local_y, half_len, local_yaw):
        return _wall(
            cx + local_x * cos_y - local_y * sin_y,
            cy + local_x * sin_y + local_y * cos_y,
            half_len,
            yaw + local_yaw,
        )

    # Back wall plus two arms; the mouth faces local +x.
    return [
        place(-depth / 2, 0.0, width / 2, np.pi / 2),
        place(0.0, width / 2, depth / 2, 0.0),
        place(0.0, -width / 2, depth / 2, 0.0),
    ]


def _gapped_wall(rng: np.random.Generator, size: float) -> list[list[float]]:
    """A long barrier with a single narrow opening.

    Local sensing can see the barrier but not where the gap is; a global path can.
    """
    yaw = rng.uniform(0, 2 * np.pi)
    span = rng.uniform(10.0, 18.0)
    gap = rng.uniform(1.6, 2.6)
    offset = rng.uniform(-span / 4, span / 4)
    cx = rng.uniform(8.0, size - 8.0)
    cy = rng.uniform(8.0, size - 8.0)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    segments = []
    for lo, hi in ((-span / 2, offset - gap / 2), (offset + gap / 2, span / 2)):
        if hi - lo < 0.6:
            continue
        mid = (lo + hi) / 2
        segments.append(
            _wall(cx + mid * cos_y, cy + mid * sin_y, (hi - lo) / 2, yaw)
        )
    return segments


def _dead_end_corridor(rng: np.random.Generator, size: float) -> list[list[float]]:
    """Two parallel walls closed at one end."""
    length = rng.uniform(4.0, 7.0)
    width = rng.uniform(1.8, 3.0)
    yaw = rng.uniform(0, 2 * np.pi)
    cx = rng.uniform(7.0, size - 7.0)
    cy = rng.uniform(7.0, size - 7.0)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    def place(local_x, local_y, half_len, local_yaw):
        return _wall(
            cx + local_x * cos_y - local_y * sin_y,
            cy + local_x * sin_y + local_y * cos_y,
            half_len,
            yaw + local_yaw,
        )

    return [
        place(0.0, width / 2, length / 2, 0.0),
        place(0.0, -width / 2, length / 2, 0.0),
        place(-length / 2, 0.0, width / 2, np.pi / 2),
    ]


def generate_maze(
    rng: np.random.Generator,
    size: float,
    cell_size: float = 3.0,
    braid_fraction: float = 0.08,
    wall_thickness: float = 0.2,
) -> np.ndarray:
    """Generate a grid maze of rectangular corridors, as in the paper's figures.

    A randomized depth-first carve produces a *perfect* maze -- exactly one route
    between any two cells. That is the wrong object here: with a unique route, a
    "suboptimal" reference path cannot exist and there is nothing to shortcut, so
    the paper's central mechanism has no way to express itself.

    So the maze is then **braided**: a fraction of interior walls is removed,
    creating loops. Now multiple routes of differing length connect any two points
    -- suboptimal paths are genuinely improvable, shortcuts genuinely exist, and
    going the wrong way still costs a long detour.

    Collinear runs are merged into single boxes. A 10x10 maze has ~180 candidate
    walls; casting rays against each individually would dominate the step cost,
    and merging typically cuts the count by two to three times.

    Returns (W, 5) walls as (cx, cy, half_length, half_thickness, yaw).
    """
    n = max(2, int(round(size / cell_size)))
    cell = size / n

    # vertical[i][j]: wall between cell (i, j) and (i, j+1)
    # horizontal[i][j]: wall between cell (i, j) and (i+1, j)
    vertical = np.ones((n, n - 1), dtype=bool)
    horizontal = np.ones((n - 1, n), dtype=bool)

    # --- randomized depth-first carve ---
    visited = np.zeros((n, n), dtype=bool)
    start = (int(rng.integers(n)), int(rng.integers(n)))
    stack = [start]
    visited[start] = True

    while stack:
        i, j = stack[-1]
        neighbours = []
        if i > 0 and not visited[i - 1, j]:
            neighbours.append((i - 1, j, "h", i - 1, j))
        if i < n - 1 and not visited[i + 1, j]:
            neighbours.append((i + 1, j, "h", i, j))
        if j > 0 and not visited[i, j - 1]:
            neighbours.append((i, j - 1, "v", i, j - 1))
        if j < n - 1 and not visited[i, j + 1]:
            neighbours.append((i, j + 1, "v", i, j))

        if not neighbours:
            stack.pop()
            continue

        ni, nj, kind, wi, wj = neighbours[rng.integers(len(neighbours))]
        if kind == "h":
            horizontal[wi, wj] = False
        else:
            vertical[wi, wj] = False
        visited[ni, nj] = True
        stack.append((ni, nj))

    # --- braid: punch loops so alternative routes exist ---
    for grid in (vertical, horizontal):
        present = np.argwhere(grid)
        if len(present) == 0:
            continue
        n_remove = int(len(present) * braid_fraction)
        if n_remove:
            for idx in rng.choice(len(present), size=n_remove, replace=False):
                grid[tuple(present[idx])] = False

    # --- merge collinear runs into single boxes ---
    walls: list[list[float]] = []
    half_thick = wall_thickness / 2

    for j in range(n - 1):                      # vertical walls at x = (j+1)*cell
        i = 0
        while i < n:
            if not vertical[i, j]:
                i += 1
                continue
            run_start = i
            while i < n and vertical[i, j]:
                i += 1
            length = (i - run_start) * cell
            walls.append([
                (j + 1) * cell,
                (run_start * cell) + length / 2,
                length / 2,
                half_thick,
                np.pi / 2,
            ])

    for i in range(n - 1):                      # horizontal walls at y = (i+1)*cell
        j = 0
        while j < n:
            if not horizontal[i, j]:
                j += 1
                continue
            run_start = j
            while j < n and horizontal[i, j]:
                j += 1
            length = (j - run_start) * cell
            walls.append([
                (run_start * cell) + length / 2,
                (i + 1) * cell,
                length / 2,
                half_thick,
                0.0,
            ])

    if not walls:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(walls, dtype=np.float32)


def generate_walls(
    rng: np.random.Generator, size: float, n_structures: tuple[int, int]
) -> np.ndarray:
    """Place a few trap structures so that greedy navigation genuinely fails."""
    builders = (_u_trap, _gapped_wall, _dead_end_corridor)
    walls: list[list[float]] = []
    for _ in range(int(rng.integers(*n_structures))):
        walls.extend(builders[rng.integers(len(builders))](rng, size))
    if not walls:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(walls, dtype=np.float32)


def _inflated_free(obstacles: np.ndarray, walls: np.ndarray, size: float, res: float) -> np.ndarray:
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

    if len(walls):
        points = np.stack([cx.ravel(), cy.ravel()], axis=1)
        clearance = wall_signed_distance(points, walls).min(axis=1)
        free &= (clearance > ROBOT_RADIUS).reshape(free.shape)

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
    n_obstacles: tuple[int, int] = (0, 8),
    radius: tuple[float, float] = (0.25, 0.60),
    n_goals: int = 8,
    n_starts: int = 12,
    min_goal_dist: float = 25.0,
    n_structures: tuple[int, int] = (2, 5),
    min_detour_ratio: float = 1.25,
    use_maze: bool = True,
    maze_cell_size: float = 3.0,
    maze_braid_fraction: float = 0.08,
    _attempt: int = 0,
) -> MapData:
    """Generate one arena with candidate start/goal pairs and geodesic fields.

    Two filters decide whether a (start, goal) pair is usable, and the second is
    what makes the benchmark meaningful:

    * geodesic separation of at least ``min_goal_dist`` -- the episode is genuinely
      long-range rather than a short hop;
    * a **detour ratio** (geodesic / straight-line) of at least
      ``min_detour_ratio`` -- reaching the goal *requires* going around something.

    Without the detour filter, "drive toward the goal and steer around obstacles"
    solves nearly every episode, a reference path contributes nothing, and the
    effect the paper is about cannot be measured. Measured on scattered convex
    obstacles alone, a policy trained only on optimal paths scored 0.96-1.00 on
    every condition including no-path-at-all -- proof the path was not being used
    because it was not needed.
    """
    def retry():
        # Structures can seal the arena; back off rather than loop forever.
        relaxed = max(1.0, min_detour_ratio - 0.05) if _attempt > 3 else min_detour_ratio
        return generate_map(
            rng, size, n_obstacles, radius, n_goals, n_starts, min_goal_dist,
            n_structures, relaxed, use_maze, maze_cell_size, maze_braid_fraction,
            _attempt + 1,
        )
    m = int(rng.integers(*n_obstacles))
    obstacles = np.stack(
        [
            rng.uniform(2.0, size - 2.0, m),
            rng.uniform(2.0, size - 2.0, m),
            rng.uniform(radius[0], radius[1], m),
        ],
        axis=1,
    ).astype(np.float32)

    if use_maze:
        walls = generate_maze(rng, size, maze_cell_size, maze_braid_fraction)
    else:
        walls = generate_walls(rng, size, n_structures)

    free = _inflated_free(obstacles, walls, size, RES)
    graph, idx = _grid_graph(free, RES)
    n_free = int(free.sum())
    if n_free < 500:
        return retry()

    keep = _largest_component(graph, n_free)
    node_rc = np.argwhere(free)                 # (n_free, 2) row/col per node
    comp_rc = node_rc[keep]                     # cells inside the main component
    if len(comp_rc) < 400:
        return retry()

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
    #
    # Starts are *selected*, not merely sampled. A large pool is scored by how many
    # goals it forms a genuinely detour-requiring pair with, and the best are kept.
    # Uniform sampling puts most starts in open space where the straight line is
    # already almost optimal, and the detour filter then rejects nearly everything
    # and falls back to accepting anything -- which is how this benchmark ended up
    # solvable without a path in the first place.
    start_pool = np.flatnonzero(keep)
    n_candidates = min(start_pool.size, max(300, n_starts * 20))
    candidate_nodes = rng.choice(start_pool, size=n_candidates, replace=False)
    candidate_xy = cell_to_world(node_rc[candidate_nodes]).astype(np.float32)

    geodesic_all = dist_flat[:, candidate_nodes]                       # (G, C)
    straight_all = np.hypot(
        candidate_xy[None, :, 0] - goals[:, None, 0],
        candidate_xy[None, :, 1] - goals[:, None, 1],
    )
    detour_all = geodesic_all / np.maximum(straight_all, 1e-6)
    qualifies = (
        np.isfinite(geodesic_all)
        & (geodesic_all >= min_goal_dist)
        & (detour_all >= min_detour_ratio)
    )
    # Rank by how many goals a candidate serves, breaking ties on mean detour.
    finite_detour = np.where(np.isfinite(detour_all), detour_all, 0.0)
    score = qualifies.sum(axis=0) + 0.01 * finite_detour.mean(axis=0)
    best = np.argsort(-score)[:n_starts]
    start_nodes = candidate_nodes[best]
    starts = candidate_xy[best]

    # A (goal, start) pair is usable only if the start can actually reach the goal
    # and is far enough away for the episode to be a long-range problem.
    starts_valid = np.zeros((n_goals, n_starts), dtype=bool)
    for k in range(n_goals):
        geodesic = dist_flat[k][start_nodes]
        straight = np.hypot(starts[:, 0] - goals[k, 0], starts[:, 1] - goals[k, 1])
        detour = geodesic / np.maximum(straight, 1e-6)

        reachable = np.isfinite(geodesic)
        starts_valid[k] = (
            reachable & (geodesic >= min_goal_dist) & (detour >= min_detour_ratio)
        )
        if not starts_valid[k].any():  # sparse map: relax rather than discard
            starts_valid[k] = reachable & (geodesic >= 0.5 * min_goal_dist)
        if not starts_valid[k].any():
            starts_valid[k] = reachable

    if not starts_valid.any():
        return retry()

    return MapData(
        obstacles=obstacles,
        walls=walls,
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
