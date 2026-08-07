"""Vectorized path-conditioned navigation environment (PyTorch backend).

Dynamics, ray casting, reward shaping and episode resets all run as batched
tensor operations on a single device. On Apple Silicon this keeps an entire
rollout resident on the GPU alongside the policy, so a PPO iteration performs no
host/device transfers except when episodes terminate and new reference paths must
be planned on the CPU.

Action space (2-D, nonholonomic wheeled base)
    action[0] -> normalized forward velocity command
    action[1] -> normalized yaw-rate command

The paper's B2W is a *wheeled quadruped* emitting (v_x, v_y, omega). A purely
wheeled robot cannot strafe, so the lateral channel is dropped; obstacle
avoidance must be solved by steering rather than sidestepping.

Observation is deliberately split, which is what makes the actor-critic
asymmetric:
    obs       -- actor input: ray scan, goal in body frame, proprioception, and
                 the *possibly corrupted* reference path.
    priv      -- critic only: true geodesic distance-to-goal, clearance, route
                 detour factor, episode phase.
    opt_path  -- critic only: the *true optimal* path, supplied even when the
                 actor was handed a corrupted or absent one.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import (
    ACTION_DIM,
    GOAL_RADIUS_M,
    GRID_RESOLUTION_M,
    MAX_ANGULAR_ACCEL,
    MAX_EPISODE_STEPS,
    MAX_FORWARD_SPEED,
    MAX_LINEAR_ACCEL,
    MAX_PATH_VERTICES,
    MAX_REVERSE_SPEED,
    MAX_YAW_RATE,
    NUM_RAYS,
    NUM_WAYPOINTS,
    PATH_VERTEX_SPACING_M,
    ROBOT_RADIUS_M,
    SENSOR_FOV_RAD,
    SENSOR_MAX_RANGE_M,
    SIM_TIMESTEP,
    SUBSTEPS_PER_ACTION,
    WAYPOINT_SPACING_M,
    EnvConfig,
)
from ..maps import MapData, generate_map_set
from ..path_library import load_or_build
from ..planning import PathQuality


class PathConditionedNavEnv:
    """Batched navigation environment. All public tensors live on ``config.device``."""

    def __init__(self, config: EnvConfig, maps: list[MapData] | None = None):
        self.config = config
        self.device = torch.device(config.device)
        self.rng = np.random.default_rng(config.seed)
        self.num_envs = config.num_envs
        self.arena_size = config.maps.arena_size_m

        self.maps = maps if maps is not None else generate_map_set(
            config.maps.num_maps,
            seed=config.seed,
            size=config.maps.arena_size_m,
            n_obstacles=config.maps.num_obstacles,
            radius=config.maps.obstacle_radius_m,
            n_goals=config.maps.goals_per_map,
            n_starts=config.maps.starts_per_goal,
            min_goal_dist=config.maps.min_start_goal_geodesic_m,
            n_structures=config.maps.num_structures,
            min_detour_ratio=config.maps.min_detour_ratio,
            use_maze=config.maps.use_maze,
            maze_cell_size=config.maps.maze_cell_size_m,
            maze_braid_fraction=config.maps.maze_braid_fraction,
        )
        self._pack_maps_to_device()
        self._load_path_library()
        self._allocate_state()
        self.reset_all()

    # ------------------------------------------------------------------ setup

    def _pack_maps_to_device(self) -> None:
        """Pack the CPU map pool into padded, indexable GPU tensors."""
        num_maps = len(self.maps)
        max_obstacles = max(len(m.obstacles) for m in self.maps)

        # Padding rows carry radius = -1 so they miss every ray and every
        # collision test without needing a separate validity mask.
        packed = np.zeros((num_maps, max_obstacles, 3), dtype=np.float32)
        packed[..., 2] = -1.0
        for i, map_data in enumerate(self.maps):
            packed[i, : len(map_data.obstacles)] = map_data.obstacles
        self.obstacles = torch.from_numpy(packed).to(self.device)

        # Walls are oriented boxes: (cx, cy, half_length, half_thickness, yaw).
        # Padding rows carry half_length = -1 and are rejected by every test.
        max_walls = max(1, max(len(m.walls) for m in self.maps))
        packed_walls = np.zeros((num_maps, max_walls, 5), dtype=np.float32)
        packed_walls[..., 2] = -1.0
        for i, map_data in enumerate(self.maps):
            if len(map_data.walls):
                packed_walls[i, : len(map_data.walls)] = map_data.walls
        self.walls = torch.from_numpy(packed_walls).to(self.device)

        geodesic = np.stack([m.dist for m in self.maps])          # (M, K, H, W)
        self.goals_per_map, self.grid_h, self.grid_w = geodesic.shape[1:]
        # Unreachable cells are +inf; clamp so gathers stay finite everywhere.
        geodesic = np.nan_to_num(geodesic, posinf=1e4)
        self.geodesic_field = torch.from_numpy(
            geodesic.reshape(num_maps * self.goals_per_map, -1)
        ).to(self.device)

        self.candidate_goals = torch.from_numpy(
            np.stack([m.goals for m in self.maps])
        ).to(self.device)                                          # (M, G, 2)
        self.candidate_starts = torch.from_numpy(
            np.stack([m.starts for m in self.maps])
        ).to(self.device)                                          # (M, S, 2)
        self.num_starts = self.candidate_starts.shape[1]

        # Flatten each map's usable (goal, start) combinations into a padded table
        # so that sampling an episode is one gather instead of a rejection loop.
        valid = np.stack([m.starts_valid for m in self.maps])       # (M, G, S)
        pair_counts = valid.reshape(num_maps, -1).sum(axis=1)
        max_pairs = int(pair_counts.max())
        pair_table = np.zeros((num_maps, max_pairs, 2), dtype=np.int64)
        for i in range(num_maps):
            goal_idx, start_idx = np.nonzero(valid[i])
            pair_table[i, : len(goal_idx), 0] = goal_idx
            pair_table[i, : len(goal_idx), 1] = start_idx
        self.valid_pairs = torch.from_numpy(pair_table).to(self.device)
        self.valid_pair_counts = torch.from_numpy(pair_counts.astype(np.int64)).to(self.device)

    def _load_path_library(self) -> None:
        """Load or build the precomputed route tables and move them on-device."""
        library = load_or_build(
            self.maps, self.config.maps, self.config.seed, self.rng
        )
        self.library_optimal = torch.from_numpy(library.optimal).to(self.device)
        self.library_optimal_len = torch.from_numpy(library.optimal_len).long().to(self.device)
        self.library_suboptimal = torch.from_numpy(library.suboptimal).to(self.device)
        self.library_suboptimal_len = (
            torch.from_numpy(library.suboptimal_len).long().to(self.device)
        )
        self.goals_per_map = self.library_optimal.shape[1]

    def _allocate_state(self) -> None:
        n, dev = self.num_envs, self.device
        zeros = lambda *shape: torch.zeros(*shape, device=dev)

        # --- robot state ---
        self.position = zeros(n, 2)
        self.heading = zeros(n)
        self.forward_speed = zeros(n)
        self.yaw_rate = zeros(n)
        self.previous_action = zeros(n, ACTION_DIM)

        # --- episode bookkeeping ---
        self.step_count = torch.zeros(n, dtype=torch.long, device=dev)
        self.map_index = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal_index = torch.zeros(n, dtype=torch.long, device=dev)
        self.goal_position = zeros(n, 2)
        self.path_quality = torch.zeros(n, dtype=torch.long, device=dev)

        # --- reference paths: what the actor sees, and the ground truth ---
        self.reference_path = zeros(n, MAX_PATH_VERTICES, 2)
        self.reference_path_len = torch.zeros(n, dtype=torch.long, device=dev)
        self.optimal_path = zeros(n, MAX_PATH_VERTICES, 2)
        self.optimal_path_len = torch.zeros(n, dtype=torch.long, device=dev)

        # --- reward memory (previous-step quantities) ---
        self.previous_geodesic = zeros(n)
        self.previous_arclength = zeros(n)

        # --- shortcut-reward bookkeeping (episode-cumulative, see step()) ---
        self.initial_geodesic = zeros(n)
        self.initial_arclength = zeros(n)
        self.best_lead = zeros(n)

        # Device-side RNG so episode resets never touch the host.
        self.generator = torch.Generator(device=dev)
        self.generator.manual_seed(self.config.seed)

        weights = torch.zeros(len(PathQuality), device=dev)
        for name, weight in self.config.path_quality_mix.items():
            weights[int(PathQuality[name])] = weight
        self._quality_weights = weights / weights.sum()
        self._offset_control_points = 16

        self.ray_bearings = torch.linspace(
            -SENSOR_FOV_RAD / 2, SENSOR_FOV_RAD / 2, NUM_RAYS, device=dev
        )
        self._vertex_indices = torch.arange(MAX_PATH_VERTICES, device=dev)[None, :]

    # ------------------------------------------------------------------ reset

    def _sample_path_qualities(self, count: int) -> torch.Tensor:
        """Draw reference-path conditions from the training mixture, or a pinned one."""
        if self.config.fixed_path_quality is not None:
            fixed = int(PathQuality[self.config.fixed_path_quality])
            return torch.full((count,), fixed, dtype=torch.long, device=self.device)
        return torch.multinomial(
            self._quality_weights, count, replacement=True, generator=self.generator
        )

    def _smooth_offsets(self, count: int, max_offset: float) -> torch.Tensor:
        """Generate smooth random path offsets on-device.

        A few random control points are linearly upsampled to the full polyline
        length, so the displacement varies gradually along the path. IID per-vertex
        noise would instead produce a zigzag no real planner would ever emit, and
        would teach the policy to reject the path for the wrong reason.
        """
        control = torch.empty(
            count, 2, self._offset_control_points, device=self.device
        ).uniform_(-1.0, 1.0)
        upsampled = torch.nn.functional.interpolate(
            control, size=MAX_PATH_VERTICES, mode="linear", align_corners=True
        ).transpose(1, 2)                                        # (count, P, 2)
        magnitude = upsampled.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return upsampled / magnitude * magnitude.clamp(max=1.0) * max_offset

    def reset_all(self) -> None:
        self._reset_envs(torch.arange(self.num_envs, device=self.device))

    def _reset_envs(self, env_ids: torch.Tensor) -> None:
        """Re-roll the given environments onto fresh map/start/goal/path draws.

        Entirely on-device: routes come from the precomputed library and the
        corrupted variants are synthesized here, so an episode reset costs a few
        gathers rather than two graph searches.
        """
        count = int(env_ids.numel())
        if count == 0:
            return

        gen = self.generator
        map_ids = torch.randint(0, len(self.maps), (count,), device=self.device, generator=gen)

        # Sample a usable (goal, start) combination from this map's pair table.
        pair_slot = (
            torch.rand(count, device=self.device, generator=gen)
            * self.valid_pair_counts[map_ids].float()
        ).long().clamp(min=0)
        pairs = self.valid_pairs[map_ids, pair_slot]
        goal_ids, start_ids = pairs[:, 0], pairs[:, 1]
        qualities = self._sample_path_qualities(count)

        starts = self.candidate_starts[map_ids, start_ids]
        goals = self.candidate_goals[map_ids, goal_ids]

        optimal = self.library_optimal[map_ids, goal_ids, start_ids]
        optimal_len = self.library_optimal_len[map_ids, goal_ids, start_ids]
        suboptimal = self.library_suboptimal[map_ids, goal_ids, start_ids]
        suboptimal_len = self.library_suboptimal_len[map_ids, goal_ids, start_ids]

        # WRONG_GOAL: a well-formed route from this same start to a different goal.
        goal_offset = torch.randint(
            1, self.goals_per_map, (count,), device=self.device, generator=gen
        )
        wrong_goal_ids = (goal_ids + goal_offset) % self.goals_per_map
        wrong = self.library_optimal[map_ids, wrong_goal_ids, start_ids]
        wrong_len = self.library_optimal_len[map_ids, wrong_goal_ids, start_ids]

        noisy = optimal + self._smooth_offsets(count, max_offset=1.0)
        detoured = suboptimal + self._smooth_offsets(count, max_offset=2.0)

        # Select per-environment according to the sampled condition.
        q = qualities[:, None, None]
        reference = torch.where(q == int(PathQuality.OPTIMAL), optimal, torch.zeros_like(optimal))
        reference = torch.where(q == int(PathQuality.NOISY), noisy, reference)
        reference = torch.where(q == int(PathQuality.SUBOPTIMAL), suboptimal, reference)
        reference = torch.where(q == int(PathQuality.DETOURED), detoured, reference)
        reference = torch.where(q == int(PathQuality.WRONG_GOAL), wrong, reference)

        lengths = torch.zeros_like(optimal_len)
        for quality, source in (
            (PathQuality.OPTIMAL, optimal_len),
            (PathQuality.NOISY, optimal_len),
            (PathQuality.SUBOPTIMAL, suboptimal_len),
            (PathQuality.DETOURED, suboptimal_len),
            (PathQuality.WRONG_GOAL, wrong_len),
        ):
            lengths = torch.where(qualities == int(quality), source, lengths)
        # PathQuality.NONE keeps length 0, which the encoder reads as "no path".

        self.map_index[env_ids] = map_ids
        self.goal_index[env_ids] = goal_ids
        self.path_quality[env_ids] = qualities
        self.position[env_ids] = starts
        self.goal_position[env_ids] = goals
        self.reference_path[env_ids] = reference
        self.reference_path_len[env_ids] = lengths
        self.optimal_path[env_ids] = optimal
        self.optimal_path_len[env_ids] = optimal_len

        # Start facing roughly toward the goal, with enough slack that the policy
        # still has to orient itself rather than driving straight off the line.
        to_goal = self.goal_position[env_ids] - self.position[env_ids]
        self.heading[env_ids] = torch.atan2(to_goal[:, 1], to_goal[:, 0]) + torch.empty(
            count, device=self.device
        ).uniform_(-np.pi / 2, np.pi / 2)

        self.forward_speed[env_ids] = 0.0
        self.yaw_rate[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.step_count[env_ids] = 0
        self.previous_geodesic[env_ids] = self._geodesic_distance(env_ids)
        self.previous_arclength[env_ids] = self._project_onto_path(
            self.reference_path, self.reference_path_len, env_ids
        )[0]
        self.initial_geodesic[env_ids] = self.previous_geodesic[env_ids]
        self.initial_arclength[env_ids] = self.previous_arclength[env_ids]
        self.best_lead[env_ids] = 0.0

    # ------------------------------------------------------------- geometry

    def _geodesic_distance(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """True geodesic distance-to-goal, gathered from the precomputed field.

        Privileged: never enters the actor's observation. Used for the progress
        and shortcut rewards and for the critic.
        """
        position = self.position if env_ids is None else self.position[env_ids]
        map_ids = self.map_index if env_ids is None else self.map_index[env_ids]
        goal_ids = self.goal_index if env_ids is None else self.goal_index[env_ids]

        cell = (position / GRID_RESOLUTION_M).long()
        col = cell[:, 0].clamp(0, self.grid_w - 1)
        row = cell[:, 1].clamp(0, self.grid_h - 1)
        field_index = map_ids * self.goals_per_map + goal_ids
        return self.geodesic_field[field_index, row * self.grid_w + col]

    def _project_onto_path(
        self,
        path: torch.Tensor,
        path_len: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the robot onto a path, returning (arclength, vertex index).

        Because polylines are resampled to uniform spacing upstream, arclength is
        simply ``index * PATH_VERTEX_SPACING_M`` -- so the projection reduces to a
        single batched argmin instead of a per-segment geometric solve.
        """
        position = self.position if env_ids is None else self.position[env_ids]
        path = path if env_ids is None else path[env_ids]
        path_len = path_len if env_ids is None else path_len[env_ids]

        sq_dist = (path - position[:, None, :]).pow(2).sum(-1)
        in_range = self._vertex_indices[:, : path.shape[1]] < path_len[:, None]
        sq_dist = torch.where(in_range, sq_dist, torch.full_like(sq_dist, float("inf")))

        vertex = sq_dist.argmin(dim=1)
        vertex = torch.where(path_len > 0, vertex, torch.zeros_like(vertex))
        return vertex.float() * PATH_VERTEX_SPACING_M, vertex

    def _cast_rays(self) -> torch.Tensor:
        """Batched ray casting against circular obstacles and the arena walls.

        Shapes broadcast as (envs, rays, obstacles); with 4096 envs and 64 rays
        this is a few million elements, which the GPU handles in one fused pass.
        """
        bearings = self.heading[:, None] + self.ray_bearings[None, :]   # (E, R)
        dir_x, dir_y = torch.cos(bearings), torch.sin(bearings)

        obstacles = self.obstacles[self.map_index]                       # (E, M, 3)
        center_x, center_y, radius = obstacles.unbind(-1)
        offset_x = self.position[:, 0:1] - center_x                      # (E, M)
        offset_y = self.position[:, 1:2] - center_y
        inflated = radius + ROBOT_RADIUS_M

        # Ray-circle intersection: |o + t*d - c|^2 = r^2 with |d| = 1, so
        # t^2 + 2*b*t + c = 0 and the near root is t = -b - sqrt(b^2 - c).
        b = offset_x[:, None, :] * dir_x[..., None] + offset_y[:, None, :] * dir_y[..., None]
        c = (offset_x**2 + offset_y**2 - inflated**2)[:, None, :]
        discriminant = b * b - c
        hits = (discriminant > 0) & (inflated[:, None, :] > 0)
        near_root = -b - torch.sqrt(discriminant.clamp(min=0))
        near_root = torch.where(
            hits & (near_root > 0), near_root, torch.full_like(near_root, SENSOR_MAX_RANGE_M)
        )
        obstacle_range = near_root.min(dim=2).values                     # (E, R)

        # Arena walls via the slab method; the robot is always inside the box.
        eps = 1e-6
        pos_x, pos_y = self.position[:, 0:1], self.position[:, 1:2]
        limit = self.arena_size - ROBOT_RADIUS_M
        t_x = torch.where(dir_x > 0, (limit - pos_x) / (dir_x + eps),
                          (ROBOT_RADIUS_M - pos_x) / (dir_x - eps))
        t_y = torch.where(dir_y > 0, (limit - pos_y) / (dir_y + eps),
                          (ROBOT_RADIUS_M - pos_y) / (dir_y - eps))
        wall_range = torch.minimum(t_x.abs(), t_y.abs())

        structure_range = self._cast_rays_at_walls(dir_x, dir_y)
        nearest = torch.minimum(torch.minimum(obstacle_range, wall_range), structure_range)
        return nearest.clamp(0.0, SENSOR_MAX_RANGE_M)

    def _cast_rays_at_walls(self, dir_x: torch.Tensor, dir_y: torch.Tensor) -> torch.Tensor:
        """Ray casting against oriented-box walls, via the slab method.

        The ray is transformed into each box's local frame (where the box is
        axis-aligned) and intersected with two slabs. Working in the local frame is
        what keeps rotated geometry as cheap as axis-aligned geometry.
        """
        walls = self.walls[self.map_index]                               # (E, W, 5)
        centre_x, centre_y, half_len, half_thick, yaw = walls.unbind(-1)
        is_real = half_len > 0
        # Inflate to configuration space, matching the circular obstacles.
        half_len = half_len + ROBOT_RADIUS_M
        half_thick = half_thick + ROBOT_RADIUS_M

        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)                    # (E, W)
        offset_x = self.position[:, 0:1] - centre_x
        offset_y = self.position[:, 1:2] - centre_y
        local_ox = (offset_x * cos_y + offset_y * sin_y)[:, None, :]     # (E, 1, W)
        local_oy = (-offset_x * sin_y + offset_y * cos_y)[:, None, :]

        local_dx = dir_x[..., None] * cos_y[:, None, :] + dir_y[..., None] * sin_y[:, None, :]
        local_dy = -dir_x[..., None] * sin_y[:, None, :] + dir_y[..., None] * cos_y[:, None, :]

        eps = 1e-9
        inv_dx = 1.0 / torch.where(local_dx.abs() < eps, torch.full_like(local_dx, eps), local_dx)
        inv_dy = 1.0 / torch.where(local_dy.abs() < eps, torch.full_like(local_dy, eps), local_dy)

        tx1 = (-half_len[:, None, :] - local_ox) * inv_dx
        tx2 = (half_len[:, None, :] - local_ox) * inv_dx
        ty1 = (-half_thick[:, None, :] - local_oy) * inv_dy
        ty2 = (half_thick[:, None, :] - local_oy) * inv_dy

        t_near = torch.maximum(torch.minimum(tx1, tx2), torch.minimum(ty1, ty2))
        t_far = torch.minimum(torch.maximum(tx1, tx2), torch.maximum(ty1, ty2))

        # Use the exit point when the origin is already inside the slab overlap.
        t = torch.where(t_near > 0, t_near, t_far)
        hit = is_real[:, None, :] & (t_far >= t_near) & (t > 0)
        return torch.where(hit, t, torch.full_like(t, SENSOR_MAX_RANGE_M)).min(dim=2).values


    def _clearance(self) -> torch.Tensor:
        """Signed distance from the robot's edge to the nearest obstacle surface.

        Negative means the footprint is overlapping something, i.e. a collision.
        """
        obstacles = self.obstacles[self.map_index]
        center_x, center_y, radius = obstacles.unbind(-1)
        surface_dist = torch.sqrt(
            (self.position[:, 0:1] - center_x) ** 2 + (self.position[:, 1:2] - center_y) ** 2
        ) - radius
        surface_dist = torch.where(
            radius > 0, surface_dist, torch.full_like(surface_dist, 1e3)
        ).min(dim=1).values

        boundary_dist = torch.minimum(
            self.position.min(dim=1).values,
            (self.arena_size - self.position).min(dim=1).values,
        )
        return torch.minimum(
            torch.minimum(surface_dist, boundary_dist), self._wall_clearance()
        ) - ROBOT_RADIUS_M

    def _wall_clearance(self) -> torch.Tensor:
        """Distance from the robot centre to the nearest oriented-box wall."""
        walls = self.walls[self.map_index]
        centre_x, centre_y, half_len, half_thick, yaw = walls.unbind(-1)
        offset_x = self.position[:, 0:1] - centre_x
        offset_y = self.position[:, 1:2] - centre_y
        cos_y, sin_y = torch.cos(-yaw), torch.sin(-yaw)

        local_x = (offset_x * cos_y - offset_y * sin_y).abs() - half_len
        local_y = (offset_x * sin_y + offset_y * cos_y).abs() - half_thick
        outside = torch.hypot(local_x.clamp(min=0.0), local_y.clamp(min=0.0))
        inside = torch.maximum(local_x, local_y).clamp(max=0.0)
        distance = outside + inside
        return torch.where(half_len > 0, distance, torch.full_like(distance, 1e3)).min(dim=1).values

    def _waypoints_in_body_frame(
        self, path: torch.Tensor, path_len: torch.Tensor, vertex: torch.Tensor
    ) -> torch.Tensor:
        """Gather the next NUM_WAYPOINTS ahead of the projection, in the body frame.

        Waypoints past the end of the path (or on an absent path) are zeroed and
        flagged invalid, so the encoder can attend only over real geometry.
        """
        stride = int(round(WAYPOINT_SPACING_M / PATH_VERTEX_SPACING_M))
        offsets = torch.arange(1, NUM_WAYPOINTS + 1, device=self.device) * stride
        indices = vertex[:, None] + offsets[None, :]
        valid = (indices < path_len[:, None]) & (path_len[:, None] > 0)
        indices = indices.clamp(max=path.shape[1] - 1)

        world = torch.gather(path, 1, indices[..., None].expand(-1, -1, 2))
        relative = world - self.position[:, None, :]
        cos_h, sin_h = torch.cos(-self.heading)[:, None], torch.sin(-self.heading)[:, None]
        body_x = relative[..., 0] * cos_h - relative[..., 1] * sin_h
        body_y = relative[..., 0] * sin_h + relative[..., 1] * cos_h

        tokens = torch.stack(
            [body_x / SENSOR_MAX_RANGE_M, body_y / SENSOR_MAX_RANGE_M, valid.float()], dim=-1
        )
        return tokens * valid[..., None]

    # ------------------------------------------------------------ observation

    def _build_observation(self) -> dict[str, torch.Tensor]:
        scan = self._cast_rays()
        _, ref_vertex = self._project_onto_path(self.reference_path, self.reference_path_len)
        reference_waypoints = self._waypoints_in_body_frame(
            self.reference_path, self.reference_path_len, ref_vertex
        )

        to_goal = self.goal_position - self.position
        cos_h, sin_h = torch.cos(-self.heading), torch.sin(-self.heading)
        goal_body_x = to_goal[:, 0] * cos_h - to_goal[:, 1] * sin_h
        goal_body_y = to_goal[:, 0] * sin_h + to_goal[:, 1] * cos_h
        goal_range = to_goal.norm(dim=1)
        goal_bearing = torch.atan2(goal_body_y, goal_body_x)

        actor_obs = torch.cat(
            [
                scan / SENSOR_MAX_RANGE_M,
                torch.stack(
                    [
                        (goal_body_x / SENSOR_MAX_RANGE_M).clamp(-3, 3),
                        (goal_body_y / SENSOR_MAX_RANGE_M).clamp(-3, 3),
                        (goal_range / self.arena_size).clamp(0, 2),
                        torch.cos(goal_bearing),
                        torch.sin(goal_bearing),
                    ],
                    dim=1,
                ),
                torch.stack(
                    [self.forward_speed / MAX_FORWARD_SPEED, self.yaw_rate / MAX_YAW_RATE], dim=1
                ),
                self.previous_action,
                (self.reference_path_len > 0).float()[:, None],
            ],
            dim=1,
        )

        # --- critic-only channels ---
        _, opt_vertex = self._project_onto_path(self.optimal_path, self.optimal_path_len)
        optimal_waypoints = self._waypoints_in_body_frame(
            self.optimal_path, self.optimal_path_len, opt_vertex
        )
        geodesic = self._geodesic_distance()
        privileged = torch.stack(
            [
                (geodesic / self.arena_size).clamp(0, 3),
                (self._clearance() / SENSOR_MAX_RANGE_M).clamp(-1, 1),
                # How much longer the true route is than the straight line -- tells
                # the critic whether a large distance-to-goal is actually hard.
                (geodesic - goal_range).clamp(0, 30) / self.arena_size,
                self.step_count.float() / MAX_EPISODE_STEPS,
            ],
            dim=1,
        )

        return {
            "obs": actor_obs,
            "path": reference_waypoints,
            "priv": privileged,
            "opt_path": optimal_waypoints,
        }

    # ------------------------------------------------------------------- step

    @staticmethod
    def action_to_velocity_commands(action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map a normalized action to (forward speed, yaw rate) commands in SI units."""
        speed_command = torch.where(
            action[:, 0] >= 0,
            action[:, 0] * MAX_FORWARD_SPEED,
            action[:, 0] * MAX_REVERSE_SPEED,
        )
        return speed_command, action[:, 1] * MAX_YAW_RATE

    def _apply_action(self, action: torch.Tensor) -> None:
        """Advance the robot state by one navigation step.

        Isolated from :meth:`step` so that the MuJoCo backend can substitute real
        rigid-body physics here while inheriting identical observation, reward and
        termination logic. Any divergence between backends is therefore *only*
        dynamics, which is exactly what the sim-to-sim experiment is measuring.
        """
        speed_command, yaw_rate_command = self.action_to_velocity_commands(action)

        # Frozen low-level controller: rate-limited velocity tracking at 50 Hz,
        # ten substeps per navigation action -- the paper's 5 Hz / 50 Hz split.
        for _ in range(SUBSTEPS_PER_ACTION):
            self.forward_speed += (speed_command - self.forward_speed).clamp(
                -MAX_LINEAR_ACCEL * SIM_TIMESTEP, MAX_LINEAR_ACCEL * SIM_TIMESTEP
            )
            self.yaw_rate += (yaw_rate_command - self.yaw_rate).clamp(
                -MAX_ANGULAR_ACCEL * SIM_TIMESTEP, MAX_ANGULAR_ACCEL * SIM_TIMESTEP
            )
            self.heading = self.heading + self.yaw_rate * SIM_TIMESTEP
            self.position = self.position + torch.stack(
                [
                    self.forward_speed * torch.cos(self.heading),
                    self.forward_speed * torch.sin(self.heading),
                ],
                dim=1,
            ) * SIM_TIMESTEP

        self.position = self.position.clamp(ROBOT_RADIUS_M, self.arena_size - ROBOT_RADIUS_M)
        self.heading = torch.atan2(torch.sin(self.heading), torch.cos(self.heading))

    def _extra_failures(self) -> torch.Tensor:
        """Backend-specific failure modes. None for idealized kinematics."""
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def step(
        self, action: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Advance every environment by one navigation step (0.2 s at 5 Hz)."""
        action = action.clamp(-1.0, 1.0)
        self._apply_action(action)
        self.step_count += 1

        geodesic = self._geodesic_distance()
        arclength, _ = self._project_onto_path(self.reference_path, self.reference_path_len)
        clearance = self._clearance()

        geodesic_progress = (self.previous_geodesic - geodesic).clamp(-5.0, 5.0)

        tipped_over = self._extra_failures()
        collided = (clearance < 0.0) | tipped_over
        reached_goal = (self.goal_position - self.position).norm(dim=1) < GOAL_RADIUS_M
        timed_out = self.step_count >= MAX_EPISODE_STEPS

        w = self.config.reward
        reward = w.progress * geodesic_progress
        reward = reward + w.time
        reward = reward + w.action_rate * (action - self.previous_action).pow(2).sum(1)
        clearance_penalty = (1.0 - (clearance / w.clearance_threshold_m).clamp(0, 1)).pow(2)
        reward = reward + w.clearance * clearance_penalty

        # Shortcut reward (the paper's novel term): fires only when the robot gains
        # more true geodesic progress than the reference-path arclength it consumed,
        # i.e. it cut a corner the path did not offer. Zero when there is no path.
        # Shortcut reward (the paper's novel term): credit for getting *ahead of*
        # the reference path -- covering more true geodesic distance than the path
        # arclength consumed to do it.
        #
        # Defined on an episode-cumulative running maximum rather than a per-step
        # delta, and that detail is the whole reward. Any one-sided bonus on a
        # signed per-step quantity is farmable by oscillation: the gains pay and
        # the losses are clamped away, so a policy can shuttle back and forth
        # collecting the positive half forever. Two successive versions of this
        # term were exploited exactly that way -- first by driving at full reverse
        # (backing up shortens path arclength faster than it costs geodesic
        # distance, since a path is always longer than the geodesic), then, after
        # gating on forward progress, by turning around and oscillating.
        #
        # A running maximum telescopes: only genuinely new ground pays, so the
        # total bonus over an episode can never exceed the lead actually achieved,
        # no matter how the robot moves in between.
        has_path = self.reference_path_len > 0
        lead = (self.initial_geodesic - geodesic) - (arclength - self.initial_arclength)
        shortcut_bonus = torch.where(
            has_path, (lead - self.best_lead).clamp(min=0.0), torch.zeros_like(lead)
        )
        self.best_lead = torch.maximum(self.best_lead, lead)
        reward = reward + w.shortcut * shortcut_bonus

        reward = reward + w.goal_bonus * reached_goal.float()
        reward = reward + w.collision * collided.float()

        done = collided | reached_goal | timed_out
        # Individual reward terms are exposed for diagnostics. "It didn't learn" is
        # one symptom with many causes; seeing which term dominates separates a
        # broken objective from a broken policy in one measurement.
        info = {
            "reward_terms": {
                "progress": w.progress * geodesic_progress,
                "shortcut": w.shortcut * shortcut_bonus,
                "clearance": w.clearance * clearance_penalty,
                "action_rate": w.action_rate * (action - self.previous_action).pow(2).sum(1),
                "time": torch.full_like(reward, w.time),
                "goal": w.goal_bonus * reached_goal.float(),
                "collision": w.collision * collided.float(),
            },
            "forward_speed": self.forward_speed.clone(),
            "success": reached_goal,
            "collision": collided,
            "tipped_over": tipped_over,
            "timeout": timed_out,
            "path_quality": self.path_quality.clone(),
            "geodesic": geodesic,
            "episode_steps": self.step_count.clone(),
        }

        self.previous_geodesic = geodesic
        self.previous_arclength = arclength
        self.previous_action = action.clone()

        if done.any():
            self._reset_envs(done.nonzero(as_tuple=True)[0])

        return self._build_observation(), reward, done, info

    @torch.no_grad()
    def observe(self) -> dict[str, torch.Tensor]:
        return self._build_observation()
