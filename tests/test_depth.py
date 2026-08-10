"""Tests for analytic depth rendering.

Structured as a ladder: camera model, then single-ray geometry against
hand-computed values, then agreement with the existing 2-D ray caster. Each rung
is verifiable without any of the rungs above it, and without any RL.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pcnav.config import ROBOT_RADIUS_M, EnvConfig, MapConfig
from pcnav.envs.torch_env import PathConditionedNavEnv
from pcnav.sim.depth import (
    DepthCameraConfig,
    camera_ray_directions,
    cast_depth,
    ray_cylinder_range,
    ray_ground_range,
    rays_to_world,
    select_nearest_obstacles,
)

# --------------------------------------------------------------------------
# 1. Camera model
# --------------------------------------------------------------------------


def test_ray_directions_are_unit_length():
    directions = camera_ray_directions(DepthCameraConfig())
    assert torch.allclose(directions.norm(dim=-1), torch.ones(directions.shape[0]), atol=1e-6)


def test_centre_pixel_of_odd_sized_image_points_straight_ahead():
    """With odd dimensions one pixel sits exactly on the optical axis."""
    config = DepthCameraConfig(width=65, height=41)
    directions = camera_ray_directions(config)
    centre = directions.reshape(config.height, config.width, 3)[
        config.height // 2, config.width // 2
    ]
    assert torch.allclose(centre, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_edge_pixels_sit_just_inside_the_field_of_view():
    """Pixel centres are inset by half a pixel, so azimuth < hfov/2 by that much."""
    config = DepthCameraConfig(width=64, height=40)
    directions = camera_ray_directions(config).reshape(config.height, config.width, 3)
    row = directions[config.height // 2]

    leftmost, rightmost = row[0], row[-1]
    azimuth_left = float(torch.atan2(leftmost[1], leftmost[0]))
    azimuth_right = float(torch.atan2(rightmost[1], rightmost[0]))

    half_fov = np.deg2rad(config.hfov_deg) / 2
    pixel_angle = np.deg2rad(config.hfov_deg) / config.width

    assert azimuth_left > 0 and azimuth_right < 0          # +y is left
    assert half_fov - pixel_angle < azimuth_left < half_fov
    assert azimuth_left == pytest.approx(-azimuth_right, abs=1e-6)


def test_image_row_ordering_is_top_down():
    """Row 0 must look up, the last row down -- a flip here silently mirrors depth."""
    config = DepthCameraConfig(mount_pitch_deg=0.0)
    directions = camera_ray_directions(config).reshape(config.height, config.width, 3)
    assert directions[0, config.width // 2, 2] > 0        # top row points up
    assert directions[-1, config.width // 2, 2] < 0       # bottom row points down


def test_world_rotation_applies_yaw_and_pitch():
    config = DepthCameraConfig(width=65, height=41)
    directions = camera_ray_directions(config)
    centre_index = (config.height // 2) * config.width + config.width // 2

    # Yaw only: the optical axis must land on the heading.
    heading = torch.tensor([0.0, np.pi / 2])
    world = rays_to_world(directions, heading, mount_pitch=0.0)
    assert torch.allclose(world[0, centre_index], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(world[1, centre_index], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)

    # Positive pitch is nose-down.
    pitched = rays_to_world(directions, torch.tensor([0.0]), mount_pitch=np.deg2rad(30.0))
    assert pitched[0, centre_index, 2] == pytest.approx(-0.5, abs=1e-6)
    assert pitched[0, centre_index, 0] == pytest.approx(np.cos(np.deg2rad(30.0)), abs=1e-6)


# --------------------------------------------------------------------------
# 2. Single-ray geometry, hand-computed
# --------------------------------------------------------------------------


def _single(origin, direction, cylinder) -> float:
    origins = torch.tensor([origin], dtype=torch.float32)
    directions = torch.tensor([[direction]], dtype=torch.float32)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    cylinders = torch.tensor([[cylinder]], dtype=torch.float32)
    return float(ray_cylinder_range(origins, directions, cylinders)[0, 0])


def test_head_on_hit():
    """Origin 5 m from a radius-1 cylinder centred at x=5: nearest surface at 4 m."""
    assert _single([0, 0, 0.5], [1, 0, 0], [5, 0, 1.0, 2.0]) == pytest.approx(4.0, abs=1e-5)


def test_ray_pointing_away_never_hits():
    assert np.isinf(_single([0, 0, 0.5], [-1, 0, 0], [5, 0, 1.0, 2.0]))


def test_ray_missing_to_the_side():
    assert np.isinf(_single([0, 0, 0.5], [1, 0, 0], [5, 3.0, 1.0, 2.0]))


def test_near_tangent_hits_and_exact_tangent_does_not():
    """Grazing rays.

    An offset just inside the radius must return a hit close to the perpendicular
    distance. An *exactly* tangent ray has a zero discriminant and is rejected --
    the standard choice in ray tracers, since a measure-zero grazing contact is
    numerically meaningless. Pinned here so it stays a decision, not an accident.
    """
    grazing = _single([0, 0.99, 0.5], [1, 0, 0], [5, 0, 1.0, 2.0])
    assert grazing == pytest.approx(5.0 - np.sqrt(1.0 - 0.99**2), abs=1e-3)
    assert np.isinf(_single([0, 1.0, 0.5], [1, 0, 0], [5, 0, 1.0, 2.0]))


def test_origin_above_a_short_cylinder_sees_through_it():
    """Being inside an obstacle's XY footprint is fine if you are above it.

    The camera sits at 0.45 m inside the circle of a 0.30 m obstacle. The near
    root is behind the camera and the far root is the exit surface, but the height
    test rejects both because the ray travels above the top -- so a horizontal ray
    correctly reports no hit while a downward ray finds the cap.
    """
    assert np.isinf(_single([5, 0, 0.45], [1, 0, 0], [5, 0, 1.0, 0.30]))
    assert _single([5, 0, 0.45], [0, 0, -1], [5, 0, 1.0, 0.30]) == pytest.approx(0.15, abs=1e-5)


def test_ray_passing_over_a_short_cylinder_misses():
    """The height test is what makes the cylinder finite rather than infinite."""
    # Rising ray reaches z = 2 at x = 4, above a cylinder only 0.5 m tall.
    assert np.isinf(_single([0, 0, 0.5], [4, 0, 2.0], [4, 0, 1.0, 0.5]))


def test_steeply_tilted_ray_uses_the_quadratic_a_term():
    """Regression guard for the single most likely bug in this module.

    Dropping `a = dx^2 + dy^2` makes tilted rays report ranges that are too short
    by exactly a factor of the XY-projection length. Here a 45-degree ray must
    return sqrt(2) times the horizontal distance.
    """
    horizontal = _single([0, 0, 0.0], [1, 0, 0], [5, 0, 1.0, 10.0])
    tilted = _single([0, 0, 0.0], [1, 0, 1], [5, 0, 1.0, 10.0])
    assert horizontal == pytest.approx(4.0, abs=1e-5)
    assert tilted == pytest.approx(4.0 * np.sqrt(2.0), abs=1e-4)


def test_padding_slots_never_produce_hits():
    assert np.isinf(_single([0, 0, 0.5], [1, 0, 0], [5, 0, -1.0, 2.0]))


def test_ground_plane_range():
    origins = torch.tensor([[0.0, 0.0, 2.0]])
    directions = torch.tensor([[[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    ranges = ray_ground_range(origins, directions)[0]
    assert ranges[0] == pytest.approx(2.0, abs=1e-6)   # straight down
    assert np.isinf(float(ranges[1]))                  # level: never meets z=0
    assert np.isinf(float(ranges[2]))                  # upward


# --------------------------------------------------------------------------
# 3. Agreement with the existing 2-D ray caster
# --------------------------------------------------------------------------


def _free_central_position(map_data, arena: float = 30.0, margin: float = 10.5) -> np.ndarray:
    """A free-space point at least `margin` from every wall.

    Two preconditions matter for the comparison below. The point must be in
    *inflated* free space, so the camera is not inside an obstacle -- there the
    2-D caster (near root only) and a full ray tracer (which returns the exit
    surface) legitimately disagree, and neither answer is physical. And it must be
    far from the walls, so wall geometry cannot contribute inside sensor range.
    """
    from pcnav.maps import cell_to_world

    cells = np.argwhere(map_data.free)
    world = cell_to_world(cells)
    inside = (
        (world[:, 0] > margin)
        & (world[:, 0] < arena - margin)
        & (world[:, 1] > margin)
        & (world[:, 1] < arena - margin)
    )
    assert inside.any(), "no free cell far enough from the walls"
    return world[inside][len(world[inside]) // 2].astype(np.float32)


def test_matches_2d_raycaster_on_horizontal_rays():
    """The strongest available check: a known-good implementation to compare to.

    Feeding the 3-D routine the exact horizontal directions the 2-D caster uses
    (so dz = 0 and a = 1) must reproduce its output.
    """
    # Arenas without trap structures: the 3-D routine under test handles cylinders,
    # so comparing against a scan that also sees oriented-box walls would be
    # measuring the missing primitive rather than the geometry. Ray-OBB support in
    # the depth module is still outstanding.
    env = PathConditionedNavEnv(
        EnvConfig(
            num_envs=8, device="cpu", seed=5,
            # Maze walls are oriented boxes, which the depth module does not yet
            # cast against.
            maps=MapConfig(num_maps=3, use_maze=False, num_structures=(0, 1)),
        )
    )
    for i in range(env.num_envs):
        env.position[i] = torch.from_numpy(
            _free_central_position(env.maps[int(env.map_index[i])])
        )
    env.heading[:] = torch.linspace(-np.pi, np.pi, env.num_envs)
    assert torch.all(env._clearance() > 0), "camera must not start inside an obstacle"

    expected = env._cast_rays()                                   # (B, 64)

    bearings = env.heading[:, None] + env.ray_bearings[None, :]
    directions = torch.stack(
        [torch.cos(bearings), torch.sin(bearings), torch.zeros_like(bearings)], dim=-1
    )
    packed = env.obstacles[env.map_index]                          # (B, M, 3)
    cylinders = torch.cat([packed, torch.full_like(packed[..., :1], 50.0)], dim=-1)
    origins = torch.cat([env.position, torch.zeros(env.num_envs, 1)], dim=1)

    actual = ray_cylinder_range(
        origins, directions, cylinders, inflation=ROBOT_RADIUS_M
    ).clamp(max=10.0)

    assert torch.allclose(actual, expected, atol=1e-3)


# --------------------------------------------------------------------------
# 4. Culling and the full pipeline
# --------------------------------------------------------------------------


def test_culling_is_exact_when_everything_fits():
    """With fewer obstacles than the quota, culling must change nothing."""
    torch.manual_seed(0)
    position = torch.rand(4, 2) * 30
    obstacles = torch.stack(
        [
            torch.rand(4, 6) * 30,
            torch.rand(4, 6) * 30,
            torch.rand(4, 6) * 0.8 + 0.3,
            torch.full((4, 6), 1.2),
        ],
        dim=-1,
    )
    config = DepthCameraConfig(width=16, height=12)
    culled = cast_depth(position, torch.zeros(4), obstacles, config, 30.0, max_obstacles=16)
    full = cast_depth(position, torch.zeros(4), obstacles, config, 30.0, max_obstacles=None)
    assert torch.allclose(culled, full, atol=1e-6)


def test_culling_ranks_by_surface_not_centre_distance():
    """A fat far-centred obstacle must outrank a thin nearer-centred one."""
    position = torch.tensor([[0.0, 0.0]])
    obstacles = torch.tensor([[[9.0, 0.0, 5.0, 2.0], [6.0, 0.0, 0.1, 2.0]]])
    kept = select_nearest_obstacles(position, obstacles, max_keep=1, max_range=10.0)
    assert float(kept[0, 0, 2]) == pytest.approx(5.0)   # the fat one, surface at 4 m


def test_ray_chunking_does_not_change_the_image():
    torch.manual_seed(1)
    position = torch.rand(3, 2) * 20 + 5
    obstacles = torch.stack(
        [
            torch.rand(3, 10) * 30,
            torch.rand(3, 10) * 30,
            torch.rand(3, 10) + 0.3,
            torch.full((3, 10), 1.5),
        ],
        dim=-1,
    )
    config = DepthCameraConfig(width=32, height=20)
    chunked = cast_depth(position, torch.zeros(3), obstacles, config, 30.0, ray_chunk=64)
    whole = cast_depth(position, torch.zeros(3), obstacles, config, 30.0, ray_chunk=None)
    assert torch.allclose(chunked, whole, atol=1e-6)


def test_empty_scene_gives_a_ground_gradient():
    """With no obstacles, depth must increase monotonically up the image."""
    config = DepthCameraConfig(mount_pitch_deg=15.0)
    empty = torch.zeros(1, 1, 4)
    empty[..., 2] = -1.0
    depth = cast_depth(
        torch.tensor([[15.0, 15.0]]), torch.zeros(1), empty, config, 30.0, max_obstacles=None
    )[0]

    column = depth[:, config.width // 2]
    assert torch.all(column[1:] - column[:-1] <= 1e-4)   # nearer at the bottom
    assert float(column[-1]) < float(column[0])
    assert torch.all(depth <= config.max_range + 1e-5)
    assert torch.all(depth > 0)


def test_output_shape_and_range():
    config = DepthCameraConfig()
    obstacles = torch.tensor([[[16.0, 15.0, 0.6, 1.4]]])
    depth = cast_depth(torch.tensor([[15.0, 15.0]]), torch.zeros(1), obstacles, config, 30.0)
    assert depth.shape == (1, config.height, config.width)
    assert torch.isfinite(depth).all()
    # A cylinder 1 m ahead must dominate the centre of the image.
    assert float(depth[0, config.height // 2, config.width // 2]) < 1.5
