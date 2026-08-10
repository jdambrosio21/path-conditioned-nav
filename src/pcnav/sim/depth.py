"""Analytic depth rendering — a batched depth image with no renderer.

A depth camera is a grid of rays with a distance reported per ray. Rather than
rasterizing geometry forward through a graphics pipeline (which needs a GPU
graphics API, and on Apple Silicon has no batched MJX/Madrona equivalent), we
shoot each pixel's ray *backwards* into the scene and solve for the nearest
intersection in closed form. Every primitive here — vertical cylinder, ground
plane, arena wall — has an analytic solution, so the whole image is a handful of
large fused tensor ops. That is precisely the kernel shape Metal runs well.


Frame conventions
-----------------
Body/camera frame: **x forward, y left, z up** (REP-103).
Image indexing: ``u`` across width (0 = leftmost column), ``v`` down height
(0 = top row).
Camera mount ``pitch`` is positive **nose-down**.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Numerical guards. `_EPS_PARALLEL` gates divisions by a ray component that may be
# zero (a perfectly horizontal ray never meets the ground plane).
_EPS_PARALLEL = 1e-8
_EPS_XY = 1e-12


@dataclass(frozen=True)
class DepthCameraConfig:
    """Intrinsics and mounting of the simulated depth camera.

    Defaults reproduce the ZED X configuration reported in the paper: 40x64 at
    105 deg horizontal / 78 deg vertical FOV, 10 m maximum range.
    """

    width: int = 64
    height: int = 40
    hfov_deg: float = 105.0
    vfov_deg: float = 78.0
    max_range: float = 10.0
    mount_height: float = 0.45   # metres above the ground plane
    mount_forward: float = 0.30  # metres ahead of the robot origin
    mount_pitch_deg: float = 10.0  # positive = tilted down

    @property
    def num_rays(self) -> int:
        return self.width * self.height


def camera_ray_directions(
    config: DepthCameraConfig,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unit ray directions in the **camera frame**, one per pixel.

    Constant for a given camera, so compute this once at startup — only the
    rotation into the world frame is per-step.

    Pixel centres are sampled (the ``+ 0.5``). Omitting that offset shifts the
    entire image by half a pixel, which is invisible in a rendering but shows up
    as a systematic bias the moment a policy learns from it.

    Returns:
        (height * width, 3) tensor, row-major (v-major, then u).
    """
    tan_h = float(np.tan(np.deg2rad(config.hfov_deg) / 2.0))
    tan_v = float(np.tan(np.deg2rad(config.vfov_deg) / 2.0))

    u = torch.arange(config.width, device=device, dtype=dtype)
    v = torch.arange(config.height, device=device, dtype=dtype)

    # Normalized image-plane offsets at unit forward distance.
    x_n = (2.0 * (u + 0.5) / config.width - 1.0) * tan_h   # + is image-right
    y_n = (2.0 * (v + 0.5) / config.height - 1.0) * tan_v  # + is image-down

    grid_y, grid_x = torch.meshgrid(y_n, x_n, indexing="ij")
    # Image-right is -y (y is left); image-down is -z (z is up).
    directions = torch.stack(
        [torch.ones_like(grid_x), -grid_x, -grid_y], dim=-1
    ).reshape(-1, 3)
    return directions / directions.norm(dim=-1, keepdim=True)


def rays_to_world(
    directions_camera: torch.Tensor,
    heading: torch.Tensor,
    mount_pitch: float,
) -> torch.Tensor:
    """Rotate camera-frame rays into the world frame.

    Args:
        directions_camera: (R, 3) constant camera-frame directions.
        heading: (B,) robot yaw in radians.
        mount_pitch: fixed camera pitch, positive nose-down.

    Returns:
        (B, R, 3) unit directions in the world frame.
    """
    cos_p, sin_p = float(np.cos(mount_pitch)), float(np.sin(mount_pitch))
    x, y, z = directions_camera.unbind(-1)
    # Pitch about +y: forward tips toward -z when pitch > 0.
    x_p = x * cos_p + z * sin_p
    z_p = -x * sin_p + z * cos_p

    cos_h = torch.cos(heading)[:, None]
    sin_h = torch.sin(heading)[:, None]
    world_x = x_p[None, :] * cos_h - y[None, :] * sin_h
    world_y = x_p[None, :] * sin_h + y[None, :] * cos_h
    world_z = z_p[None, :].expand_as(world_x)
    return torch.stack([world_x, world_y, world_z], dim=-1)


def camera_origins(
    position: torch.Tensor, heading: torch.Tensor, config: DepthCameraConfig
) -> torch.Tensor:
    """World-frame camera position, given the robot pose. Returns (B, 3)."""
    forward_x = torch.cos(heading) * config.mount_forward
    forward_y = torch.sin(heading) * config.mount_forward
    return torch.stack(
        [
            position[:, 0] + forward_x,
            position[:, 1] + forward_y,
            torch.full_like(heading, config.mount_height),
        ],
        dim=-1,
    )


def ray_cylinder_range(
    origins: torch.Tensor,
    directions: torch.Tensor,
    cylinders: torch.Tensor,
    inflation: float = 0.0,
) -> torch.Tensor:
    """Nearest intersection of each ray with a set of finite vertical cylinders.

    Args:
        origins:    (B, 3) ray origins.
        directions: (B, R, 3) unit ray directions.
        cylinders:  (B, M, 4) as (centre_x, centre_y, radius, height); the base
                    sits on z = 0. A non-positive radius marks a padding slot and
                    never produces a hit.
        inflation:  added to every radius. Real cameras see the true surface
                    (0.0); pass the robot radius only to compare against the
                    configuration-space 2-D scan.

    Returns:
        (B, R) Euclidean range to the nearest hit, ``inf`` where no hit.

    The side-surface solve substitutes the ray into the circle equation in XY,
    giving ``a t^2 + 2b t + c = 0`` with::

        a = dx^2 + dy^2
        b = fx*dx + fy*dy
        c = fx^2 + fy^2 - r^2

    **The ``a`` term is the whole difference from the 2-D case.** There, rays were
    unit vectors *in the plane* so ``a == 1`` and it dropped out. In 3-D a tilted
    ray has a shorter XY projection, ``a < 1``, and omitting it yields ranges that
    are systematically too short — worst for steeply tilted rays, and entirely
    plausible-looking.
    """
    centre_x, centre_y, radius, height = cylinders.unbind(-1)       # (B, M) each
    radius = radius + torch.where(radius > 0, inflation, 0.0)
    is_real = radius > 0

    dir_x, dir_y, dir_z = directions.unbind(-1)                      # (B, R)
    origin_x, origin_y, origin_z = origins.unbind(-1)                # (B,)

    offset_x = origin_x[:, None] - centre_x                          # (B, M)
    offset_y = origin_y[:, None] - centre_y

    a = (dir_x * dir_x + dir_y * dir_y).clamp(min=_EPS_XY)           # (B, R)
    b = offset_x[:, None, :] * dir_x[..., None] + offset_y[:, None, :] * dir_y[..., None]
    c = (offset_x**2 + offset_y**2 - radius**2)[:, None, :]          # (B, 1, M)

    discriminant = b * b - a[..., None] * c
    has_roots = (discriminant > 0) & is_real[:, None, :]
    sqrt_disc = torch.sqrt(discriminant.clamp(min=0.0))

    infinity = torch.full_like(b, float("inf"))
    base_z = origin_z[:, None, None]
    dir_z_expanded = dir_z[..., None]

    def _validate(t: torch.Tensor) -> torch.Tensor:
        """Keep only roots that are ahead of the ray and within the finite height."""
        hit_z = base_z + t * dir_z_expanded
        ok = has_roots & (t > 0) & (hit_z >= 0.0) & (hit_z <= height[:, None, :])
        return torch.where(ok, t, infinity)

    t_near = _validate((-b - sqrt_disc) / a[..., None])
    # The far root matters when the ray enters through the open top and leaves
    # through the side, so the near root fails the height test.
    t_far = _validate((-b + sqrt_disc) / a[..., None])
    t_side = torch.minimum(t_near, t_far)

    # Top cap: the disc at z = height.
    safe_dir_z = torch.where(
        dir_z_expanded.abs() < _EPS_PARALLEL,
        torch.full_like(dir_z_expanded, _EPS_PARALLEL),
        dir_z_expanded,
    )
    t_cap = (height[:, None, :] - base_z) / safe_dir_z
    cap_x = origin_x[:, None, None] + t_cap * dir_x[..., None]
    cap_y = origin_y[:, None, None] + t_cap * dir_y[..., None]
    within_disc = (cap_x - centre_x[:, None, :]) ** 2 + (
        cap_y - centre_y[:, None, :]
    ) ** 2 <= radius[:, None, :] ** 2
    cap_ok = (
        within_disc
        & (t_cap > 0)
        & is_real[:, None, :]
        & (dir_z_expanded.abs() >= _EPS_PARALLEL)
    )
    t_cap = torch.where(cap_ok, t_cap, infinity)

    return torch.minimum(t_side, t_cap).min(dim=2).values


def ray_ground_range(origins: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Range to the ground plane z = 0. ``inf`` for level or upward rays.

    Ground returns fill the lower rows of a depth image and are how the sensor
    perceives terrain shape — including negative obstacles, which a horizontal
    scan cannot see at all.
    """
    dir_z = directions[..., 2]
    origin_z = origins[:, 2][:, None]
    t = torch.where(dir_z < -_EPS_PARALLEL, -origin_z / dir_z, torch.full_like(dir_z, float("inf")))
    return torch.where(t > 0, t, torch.full_like(t, float("inf")))


def ray_wall_range(
    origins: torch.Tensor,
    directions: torch.Tensor,
    arena_size: float,
    wall_height: float = 1.0,
) -> torch.Tensor:
    """Range to the arena boundary walls, assuming the camera is inside the box.

    Rays that clear the top of a wall report ``inf`` rather than a spurious hit.
    """
    origin_x, origin_y, origin_z = origins.unbind(-1)
    dir_x, dir_y, dir_z = directions.unbind(-1)
    infinity = torch.full_like(dir_x, float("inf"))

    def _plane(origin, direction, bound: float) -> torch.Tensor:
        safe = torch.where(
            direction.abs() < _EPS_PARALLEL,
            torch.full_like(direction, _EPS_PARALLEL),
            direction,
        )
        t = (bound - origin[:, None]) / safe
        return torch.where((t > 0) & (direction.abs() >= _EPS_PARALLEL), t, infinity)

    candidates = torch.stack(
        [
            _plane(origin_x, dir_x, 0.0),
            _plane(origin_x, dir_x, arena_size),
            _plane(origin_y, dir_y, 0.0),
            _plane(origin_y, dir_y, arena_size),
        ],
        dim=0,
    )
    t = candidates.min(dim=0).values
    hit_z = origin_z[:, None] + t * dir_z
    over_the_top = hit_z > wall_height
    return torch.where(over_the_top | (hit_z < 0.0), infinity, t)


def select_nearest_obstacles(
    position: torch.Tensor, obstacles: torch.Tensor, max_keep: int, max_range: float
) -> torch.Tensor:
    """Broad-phase cull: keep only the ``max_keep`` nearest relevant obstacles.

    The naive ``(envs, rays, obstacles)`` broadcast is what makes depth casting
    expensive; most obstacles are out of range anyway. This is the same idea as
    frustum culling or BVH traversal in a real renderer, in a few lines.

    Ranking uses **surface** distance, not centre distance. A large-radius
    obstacle can be farther by centre yet nearer by surface, so ranking by centre
    silently discards the very obstacle about to fill the frame.

    Args:
        position:  (B, 2) robot xy.
        obstacles: (B, M, 4) as (cx, cy, radius, height).
    Returns:
        (B, min(max_keep, M), 4) with culled slots zeroed (radius = -1).
    """
    centre_x, centre_y, radius, _ = obstacles.unbind(-1)
    surface_distance = (
        torch.sqrt((position[:, 0:1] - centre_x) ** 2 + (position[:, 1:2] - centre_y) ** 2)
        - radius
    )
    relevant = (radius > 0) & (surface_distance < max_range)
    score = torch.where(relevant, surface_distance, torch.full_like(surface_distance, float("inf")))

    keep = min(max_keep, obstacles.shape[1])
    order = score.topk(keep, dim=1, largest=False).indices
    selected = torch.gather(obstacles, 1, order[..., None].expand(-1, -1, obstacles.shape[-1]))

    # Blank out slots that were only selected to fill the quota.
    kept_score = torch.gather(score, 1, order)
    padding = torch.isinf(kept_score)[..., None]
    blanked = selected.clone()
    blanked[..., 2] = torch.where(
        padding[..., 0], torch.full_like(selected[..., 2], -1.0), selected[..., 2]
    )
    return blanked


def cast_depth(
    position: torch.Tensor,
    heading: torch.Tensor,
    obstacles: torch.Tensor,
    config: DepthCameraConfig,
    arena_size: float,
    directions_camera: torch.Tensor | None = None,
    max_obstacles: int | None = 12,
    ray_chunk: int | None = 640,
    wall_height: float = 1.0,
    inflation: float = 0.0,
) -> torch.Tensor:
    """Render a batched depth image.

    Args:
        position:   (B, 2) robot xy.
        heading:    (B,) robot yaw.
        obstacles:  (B, M, 4) as (cx, cy, radius, height); radius <= 0 is padding.
        max_obstacles: broad-phase cull target, or None to disable.
        ray_chunk:  process this many rays at a time, or None for all at once.
                    Chunking trades a negligible amount of wall-clock for a large
                    reduction in peak memory.

    Returns:
        (B, height, width) **z-depth** in metres, clipped to ``config.max_range``.

    Note this returns z-depth (distance along the optical axis), not Euclidean
    range — that is what physical depth cameras and MuJoCo's depth buffer report.
    The two differ by up to 1/cos(hfov/2) at the image corners.
    """
    device = position.device
    if directions_camera is None:
        directions_camera = camera_ray_directions(config, device=device, dtype=position.dtype)

    if max_obstacles is not None:
        obstacles = select_nearest_obstacles(position, obstacles, max_obstacles, config.max_range)

    origins = camera_origins(position, heading, config)
    mount_pitch = float(np.deg2rad(config.mount_pitch_deg))

    num_rays = directions_camera.shape[0]
    chunk = num_rays if ray_chunk is None else min(ray_chunk, num_rays)

    ranges: list[torch.Tensor] = []
    for start in range(0, num_rays, chunk):
        block_camera = directions_camera[start : start + chunk]
        block_world = rays_to_world(block_camera, heading, mount_pitch)

        block_range = torch.minimum(
            ray_cylinder_range(origins, block_world, obstacles, inflation=inflation),
            torch.minimum(
                ray_ground_range(origins, block_world),
                ray_wall_range(origins, block_world, arena_size, wall_height),
            ),
        )
        # Range -> z-depth. The camera-frame x component is cos(angle to axis).
        axis_cosine = block_camera[:, 0][None, :]
        ranges.append((block_range * axis_cosine).clamp(max=config.max_range))

    depth = torch.cat(ranges, dim=1)
    depth = torch.nan_to_num(depth, nan=config.max_range, posinf=config.max_range)
    return depth.reshape(-1, config.height, config.width)
