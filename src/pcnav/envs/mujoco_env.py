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

# Closed-loop gains for the low-level velocity controller.
#
# Open-loop skid-steer inverse kinematics does not work: rotating a rigid
# four-wheel base requires scrubbing every wheel sideways against full friction,
# and measured tracking was 0.08 rad/s achieved for 1.0 rad/s commanded -- an 8%
# ratio that leaves the robot unable to turn a maze corner at all, and produced
# 0.00 success in MuJoCo against 0.98 under idealized dynamics.
#
# The paper's low-level controller is a *trained locomotion policy* that closes
# the loop on the velocity command. An open-loop map is not a controller, so this
# is a modelling error rather than a physics finding. A PI controller on measured
# body velocity is the honest minimal stand-in.
# Gains from a sweep over (kp, ki, chassis mass); 6.0/4.0 overshot to 1.35 rad/s
# on a 1.0 command while still stalling at 0.06 on a 0.5 command -- stiction plus
# an over-aggressive loop. These track to ~0.1 combined error.
# Disabled. The wheel actuators are already velocity servos, so an outer loop on
# body velocity stacks two integrators and oscillates: with a zero command it
# amplified settling noise into 4.7 rad/s of wheel drive and walked the robot into
# walls. With the caster lift restoring wheel traction and the yaw feedforward
# calibrated, open-loop inverse kinematics tracks to ~0.08 mean error, so the
# outer loop has nothing left to correct.
YAW_RATE_KP = 0.0
YAW_RATE_KI = 0.0
SPEED_KP = 0.0
SPEED_KI = 0.0
INTEGRAL_LIMIT = 8.0   # anti-windup clamp on the accumulated correction


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
        self._speed_integral = np.zeros(self.num_envs)
        self._yaw_integral = np.zeros(self.num_envs)
        # Controller state, held separately from the measured body velocity. These
        # are the rate-limited *setpoints* the low-level loop is chasing.
        self._speed_setpoint = np.zeros(self.num_envs)
        self._yaw_setpoint = np.zeros(self.num_envs)

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
            self._speed_integral[env_id] = 0.0
            self._yaw_integral[env_id] = 0.0
            self._speed_setpoint[env_id] = 0.0
            self._yaw_setpoint[env_id] = 0.0

    # ---------------------------------------------------------------- dynamics

    def _apply_action_mujoco(self, action: torch.Tensor) -> None:
        """Step real physics, then write the resulting pose back to the delegate."""
        speed_cmd, yaw_cmd = self.delegate.action_to_velocity_commands(action)
        dt = 1.0 / LOW_LEVEL_HZ
        substeps_per_control = LOW_LEVEL_HZ / NAV_POLICY_HZ
        control_decimation = max(1, int(round(self.physics_substeps / substeps_per_control)))

        for env_id in range(self.num_envs):
            model, data = self.models[env_id], self.datas[env_id]

            # Rate-limit the setpoint exactly as the idealized backend does, so the
            # two differ in physics rather than in the command they are given.
            # The setpoint is controller state and must persist across steps. Seeding
            # it from the measured body velocity -- which is what `delegate.
            # forward_speed` holds after write-back -- closes a positive feedback
            # loop: a physics disturbance raises the measurement, which raises the
            # setpoint, which drives harder. With a zero action the robot drifted
            # 0.37 m in four steps and eventually into a wall.
            speed_setpoint = float(self._speed_setpoint[env_id])
            yaw_setpoint = float(self._yaw_setpoint[env_id])
            target_speed = float(speed_cmd[env_id])
            target_yaw = float(yaw_cmd[env_id])
            speed_integral = self._speed_integral[env_id]
            yaw_integral = self._yaw_integral[env_id]

            for substep in range(self.physics_substeps):
                if substep % control_decimation == 0:
                    speed_setpoint += np.clip(
                        target_speed - speed_setpoint,
                        -MAX_LINEAR_ACCEL * dt,
                        MAX_LINEAR_ACCEL * dt,
                    )
                    yaw_setpoint += np.clip(
                        target_yaw - yaw_setpoint,
                        -MAX_ANGULAR_ACCEL * dt,
                        MAX_ANGULAR_ACCEL * dt,
                    )

                    # Close the loop on measured body velocity.
                    measured_speed, measured_yaw = self._measure_body_velocity(data)
                    speed_error = speed_setpoint - measured_speed
                    yaw_error = yaw_setpoint - measured_yaw
                    speed_integral = float(
                        np.clip(speed_integral + speed_error * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                    )
                    yaw_integral = float(
                        np.clip(yaw_integral + yaw_error * dt, -INTEGRAL_LIMIT, INTEGRAL_LIMIT)
                    )

                    commanded_speed = (
                        speed_setpoint + SPEED_KP * speed_error + SPEED_KI * speed_integral
                    )
                    commanded_yaw = (
                        yaw_setpoint + YAW_RATE_KP * yaw_error + YAW_RATE_KI * yaw_integral
                    )
                    data.ctrl[:] = body_velocity_to_wheel_speeds(commanded_speed, commanded_yaw)
                self.mujoco.mj_step(model, data)

            self._speed_integral[env_id] = speed_integral
            self._yaw_integral[env_id] = yaw_integral
            self._speed_setpoint[env_id] = speed_setpoint
            self._yaw_setpoint[env_id] = yaw_setpoint
            self._write_back_state(env_id)

    @staticmethod
    def _measure_body_velocity(data) -> tuple[float, float]:
        """Forward speed and yaw rate of the chassis, in the body frame."""
        quat = data.qpos[3:7]
        w, x, y, z = quat
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        world_vel = data.qvel[0:2]
        forward = float(world_vel[0] * np.cos(heading) + world_vel[1] * np.sin(heading))
        return forward, float(data.qvel[5])

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
