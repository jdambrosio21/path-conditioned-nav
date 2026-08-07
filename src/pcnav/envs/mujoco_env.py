"""MuJoCo-backed navigation environment.

Subclasses the PyTorch environment and replaces **only** the dynamics: map
generation, reference-path planning, observation construction, reward shaping and
termination are all inherited unchanged. Any performance gap between the two
backends is therefore attributable to physics alone, which is what makes the
sim-to-sim comparison meaningful.

What MuJoCo adds that the idealized backend cannot express:
  * wheel/ground contact, so commanded velocity is only approximately achieved,
  * skid-steer slip during turns,
  * suspension-free body roll and genuine tip-over,
  * real collision response against obstacle geometry.

This backend is for evaluation and visualization, not training. Each environment
carries its own compiled model and is stepped sequentially, so throughput is
orders of magnitude below the batched backend -- see README for measurements.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import (
    LOW_LEVEL_HZ,
    MAX_ANGULAR_ACCEL,
    MAX_LINEAR_ACCEL,
    NAV_POLICY_HZ,
    EnvConfig,
)
from ..maps import MapData
from ..sim.mjcf import body_velocity_to_wheel_speeds, build_scene_xml

# Beyond this body tilt the platform is considered to have rolled over.
TIP_OVER_TILT_RAD = np.deg2rad(50.0)


class MuJoCoNavEnv:
    """Thin sequential MuJoCo wrapper exposing the batched env's interface.

    Composition rather than inheritance for the MuJoCo handles themselves, but the
    environment *logic* is delegated to a :class:`PathConditionedNavEnv` instance
    running on CPU, whose robot state this class overwrites from MuJoCo each step.
    """

    def __init__(self, config: EnvConfig, maps: list[MapData] | None = None):
        import mujoco  # imported lazily so the training path never needs MuJoCo

        self.mujoco = mujoco
        from .torch_env import PathConditionedNavEnv

        # Physics must run on CPU regardless of what the policy uses.
        config = EnvConfig(**{**config.__dict__, "device": "cpu"})
        self.delegate = PathConditionedNavEnv(config, maps)
        self.config = config
        self.num_envs = config.num_envs
        self.device = torch.device("cpu")

        self.physics_substeps = 0
        self.models: list = [None] * self.num_envs
        self.datas: list = [None] * self.num_envs
        self._tipped = torch.zeros(self.num_envs, dtype=torch.bool)

        # Wire the delegate's dynamics and failure hooks to this backend.
        self.delegate._apply_action = self._apply_action_mujoco  # type: ignore[method-assign]
        self.delegate._extra_failures = lambda: self._tipped     # type: ignore[method-assign]
        original_reset = self.delegate._reset_envs

        def reset_with_physics(env_ids: torch.Tensor) -> None:
            original_reset(env_ids)
            self._rebuild_models(env_ids)

        self.delegate._reset_envs = reset_with_physics  # type: ignore[method-assign]
        self._rebuild_models(torch.arange(self.num_envs))

    # ------------------------------------------------------------------ models

    def _rebuild_models(self, env_ids: torch.Tensor) -> None:
        """Compile a fresh MJCF scene for each environment that just reset.

        A new episode may land on a different arena, so the model itself changes.
        Compilation costs a few milliseconds; at evaluation scale that is
        negligible next to the 100 physics steps each navigation step requires.
        """
        for env_id in env_ids.tolist():
            map_data = self.delegate.maps[int(self.delegate.map_index[env_id])]
            start_xy = self.delegate.position[env_id].numpy()
            heading = float(self.delegate.heading[env_id])

            xml = build_scene_xml(map_data, start_xy, heading)
            model = self.mujoco.MjModel.from_xml_string(xml)
            data = self.mujoco.MjData(model)
            self.mujoco.mj_forward(model, data)

            self.models[env_id] = model
            self.datas[env_id] = data
            self.physics_substeps = int(round(1.0 / (NAV_POLICY_HZ * model.opt.timestep)))
            self._tipped[env_id] = False

    # ---------------------------------------------------------------- dynamics

    def _apply_action_mujoco(self, action: torch.Tensor) -> None:
        """Step real physics, then write the resulting pose back to the delegate."""
        speed_cmd, yaw_cmd = self.delegate.action_to_velocity_commands(action)
        dt = 1.0 / LOW_LEVEL_HZ
        substeps_per_control = LOW_LEVEL_HZ / NAV_POLICY_HZ
        control_decimation = max(1, int(round(self.physics_substeps / substeps_per_control)))

        for env_id in range(self.num_envs):
            model, data = self.models[env_id], self.datas[env_id]

            # Rate-limit the command exactly as the idealized backend does, so the
            # two differ in physics rather than in the controller.
            speed = float(self.delegate.forward_speed[env_id])
            yaw_rate = float(self.delegate.yaw_rate[env_id])
            target_speed = float(speed_cmd[env_id])
            target_yaw = float(yaw_cmd[env_id])

            for substep in range(self.physics_substeps):
                if substep % control_decimation == 0:
                    speed += np.clip(
                        target_speed - speed, -MAX_LINEAR_ACCEL * dt, MAX_LINEAR_ACCEL * dt
                    )
                    yaw_rate += np.clip(
                        target_yaw - yaw_rate, -MAX_ANGULAR_ACCEL * dt, MAX_ANGULAR_ACCEL * dt
                    )
                    data.ctrl[:] = body_velocity_to_wheel_speeds(speed, yaw_rate)
                self.mujoco.mj_step(model, data)

            self._write_back_state(env_id)

    def _write_back_state(self, env_id: int) -> None:
        """Read the chassis pose/twist out of MuJoCo into the delegate's tensors."""
        data = self.datas[env_id]
        position = data.qpos[0:2].copy()
        quat = data.qpos[3:7].copy()          # MuJoCo order: (w, x, y, z)

        w, x, y, z = quat
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Body-frame forward speed and yaw rate from the free joint's twist.
        world_vel = data.qvel[0:2].copy()
        forward_speed = float(world_vel[0] * np.cos(heading) + world_vel[1] * np.sin(heading))
        yaw_rate = float(data.qvel[5])

        # Tip-over: angle between the chassis z-axis and world up.
        up_z = 1.0 - 2.0 * (x * x + y * y)
        tilt = float(np.arccos(np.clip(up_z, -1.0, 1.0)))

        arena = self.delegate.arena_size
        self.delegate.position[env_id] = torch.from_numpy(
            np.clip(position, 0.0, arena).astype(np.float32)
        )
        self.delegate.heading[env_id] = float(heading)
        self.delegate.forward_speed[env_id] = forward_speed
        self.delegate.yaw_rate[env_id] = yaw_rate
        self._tipped[env_id] = bool(tilt > TIP_OVER_TILT_RAD)

    # ------------------------------------------------------------------- API

    def step(self, action: torch.Tensor):
        return self.delegate.step(action)

    def observe(self) -> dict[str, torch.Tensor]:
        return self.delegate.observe()

    def reset_all(self) -> None:
        self.delegate.reset_all()

    def __getattr__(self, name: str):
        """Expose the delegate's state (position, goal_position, maps, ...)."""
        return getattr(self.__dict__["delegate"], name)
