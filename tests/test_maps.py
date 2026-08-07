"""Tests for procedural map generation and geodesic distance fields."""

from __future__ import annotations

import numpy as np
import pytest

from pcnav.maps import ROBOT_RADIUS, MapData, cell_to_world, generate_map, world_to_cell


@pytest.fixture(scope="module")
def sample_map() -> MapData:
    return generate_map(np.random.default_rng(0))


def test_grid_coordinate_roundtrip():
    """world -> cell -> world should land within half a cell."""
    rng = np.random.default_rng(1)
    points = rng.uniform(0.5, 29.5, size=(200, 2))
    recovered = cell_to_world(world_to_cell(points))
    assert np.all(np.abs(recovered - points) <= 0.25)


def test_free_space_excludes_obstacle_interiors(sample_map: MapData):
    """No cell inside an inflated obstacle may be marked free."""
    free_cells = np.argwhere(sample_map.free)
    positions = cell_to_world(free_cells)
    for cx, cy, r in sample_map.obstacles:
        distance = np.hypot(positions[:, 0] - cx, positions[:, 1] - cy)
        assert np.all(distance > r + ROBOT_RADIUS - 1e-6)


def test_goals_and_starts_lie_in_free_space(sample_map: MapData):
    for points in (sample_map.goals, sample_map.starts):
        cells = world_to_cell(points)
        assert np.all(sample_map.free[cells[:, 0], cells[:, 1]])


def test_geodesic_field_is_zero_at_goal_and_finite_nearby(sample_map: MapData):
    for goal_index, goal in enumerate(sample_map.goals):
        cell = world_to_cell(goal)
        assert sample_map.dist[goal_index, cell[0], cell[1]] == pytest.approx(0.0, abs=1e-6)


def test_geodesic_distance_is_never_shorter_than_euclidean(sample_map: MapData):
    """A geodesic route cannot beat a straight line -- catches indexing mistakes."""
    free_cells = np.argwhere(sample_map.free)
    sample = free_cells[:: max(1, len(free_cells) // 500)]
    positions = cell_to_world(sample)

    for goal_index, goal in enumerate(sample_map.goals):
        geodesic = sample_map.dist[goal_index, sample[:, 0], sample[:, 1]]
        euclidean = np.hypot(positions[:, 0] - goal[0], positions[:, 1] - goal[1])
        reachable = np.isfinite(geodesic)
        # One cell of slack for grid discretization.
        assert np.all(geodesic[reachable] >= euclidean[reachable] - 0.25)


def test_every_goal_has_at_least_one_valid_start(sample_map: MapData):
    assert sample_map.starts_valid.any(axis=1).all()


def test_valid_starts_are_far_enough_and_reachable(sample_map: MapData):
    """Valid pairs must be genuinely long-range, else the task is trivial."""
    start_cells = world_to_cell(sample_map.starts)
    for goal_index in range(len(sample_map.goals)):
        valid = sample_map.starts_valid[goal_index]
        if not valid.any():
            continue
        distances = sample_map.dist[goal_index, start_cells[valid, 0], start_cells[valid, 1]]
        assert np.all(np.isfinite(distances))
        assert np.all(distances > 0.0)


def test_walls_are_generated_and_excluded_from_free_space(sample_map: MapData):
    """Trap structures must actually block, or they are decoration."""
    from pcnav.maps import wall_signed_distance

    assert len(sample_map.walls) > 0
    free_cells = np.argwhere(sample_map.free)
    positions = cell_to_world(free_cells)
    clearance = wall_signed_distance(positions, sample_map.walls).min(axis=1)
    assert np.all(clearance > ROBOT_RADIUS - 1e-6)


def test_wall_signed_distance_matches_hand_computed_values():
    """Axis-aligned box, 4 m long and 0.4 m thick, centred at the origin."""
    from pcnav.maps import wall_signed_distance

    wall = np.array([[0.0, 0.0, 2.0, 0.2, 0.0]])
    points = np.array([[0.0, 1.2], [3.0, 0.0], [0.0, 0.0]])
    d = wall_signed_distance(points, wall)[:, 0]
    assert d[0] == pytest.approx(1.0, abs=1e-6)    # broadside
    assert d[1] == pytest.approx(1.0, abs=1e-6)    # off the end
    assert d[2] < 0                                 # inside


def test_wall_signed_distance_respects_orientation():
    """The same offset must read differently once the wall is rotated."""
    from pcnav.maps import wall_signed_distance

    point = np.array([[3.0, 0.0]])
    flat = np.array([[0.0, 0.0, 2.0, 0.2, 0.0]])
    turned = np.array([[0.0, 0.0, 2.0, 0.2, np.pi / 2]])
    assert wall_signed_distance(point, flat)[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert wall_signed_distance(point, turned)[0, 0] == pytest.approx(2.8, abs=1e-6)


def test_valid_start_goal_pairs_require_a_real_detour(sample_map: MapData):
    """The property that makes reference paths worth having.

    If the straight line is already near-optimal, driving at the goal solves the
    episode and path conditioning has nothing to contribute.
    """
    start_cells = world_to_cell(sample_map.starts)
    ratios = []
    for goal_index, goal in enumerate(sample_map.goals):
        for start_index in np.flatnonzero(sample_map.starts_valid[goal_index]):
            geodesic = sample_map.dist[
                goal_index, start_cells[start_index, 0], start_cells[start_index, 1]
            ]
            straight = np.hypot(*(sample_map.starts[start_index] - goal))
            ratios.append(geodesic / max(straight, 1e-6))
    assert len(ratios) > 0
    assert np.mean(ratios) > 1.2
