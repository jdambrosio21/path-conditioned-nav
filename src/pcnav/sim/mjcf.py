"""MJCF model generation: turn a procedural :class:`MapData` into a MuJoCo scene.

The robot is a four-wheel skid-steer base sized like a B2W. Skid steer is the
honest wheeled analogue of the paper's platform: it is nonholonomic, it slips
when turning under load, and it can be tipped by aggressive commands -- none of
which the idealized training dynamics model. That mismatch is the point, since it
is what the sim-to-sim evaluation measures.

Obstacles become cylinders and the arena is walled, matching the geometry the
PyTorch backend ray-casts against analytically, so both backends solve the
*same* navigation problem under different physics.
"""

from __future__ import annotations

import numpy as np

from ..config import ROBOT_RADIUS_M
from ..maps import MapData

# --- chassis geometry (metres, kilograms) ---
CHASSIS_HALF_LENGTH = 0.35
CHASSIS_HALF_WIDTH = 0.22
CHASSIS_HALF_HEIGHT = 0.10
CHASSIS_MASS = 40.0
WHEEL_RADIUS = 0.16
WHEEL_HALF_WIDTH = 0.06
WHEEL_MASS = 2.5
WHEEL_TRACK = 0.30      # lateral offset of each wheel from the chassis centreline
WHEEL_BASE = 0.28       # longitudinal offset of each wheel from the chassis centre
CHASSIS_RIDE_HEIGHT = WHEEL_RADIUS

OBSTACLE_HEIGHT = 1.2
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2


def _wheel_body(name: str, x: float, y: float) -> str:
    """One driven wheel: a hinge about the body-lateral axis with a velocity motor."""
    return f"""
      <body name="{name}" pos="{x:.3f} {y:.3f} 0">
        <joint name="{name}_joint" type="hinge" axis="0 1 0" damping="0.4"/>
        <geom name="{name}_geom" type="cylinder" size="{WHEEL_RADIUS} {WHEEL_HALF_WIDTH}"
              quat="0.707107 0.707107 0 0" mass="{WHEEL_MASS}"
              friction="1.4 0.02 0.001" rgba="0.15 0.15 0.17 1"/>
      </body>"""


def build_scene_xml(
    map_data: MapData,
    start_xy: np.ndarray,
    start_heading: float = 0.0,
    include_visual_markers: bool = True,
) -> str:
    """Build a complete MJCF document for one arena and one spawn pose.

    Args:
        map_data: the procedural arena to realize.
        start_xy: robot spawn position in world metres.
        start_heading: spawn yaw in radians.
        include_visual_markers: add non-colliding site markers used by the viewer
            to draw the goal and the reference path.
    """
    size = map_data.size
    half = size / 2.0

    obstacles = "\n".join(
        f'      <geom name="obstacle_{i}" type="cylinder" '
        f'pos="{cx:.3f} {cy:.3f} {OBSTACLE_HEIGHT / 2:.3f}" '
        f'size="{r:.3f} {OBSTACLE_HEIGHT / 2:.3f}" rgba="0.45 0.35 0.30 1"/>'
        for i, (cx, cy, r) in enumerate(map_data.obstacles)
    )

    walls = "\n".join(
        f'      <geom name="wall_{n}" type="box" pos="{px:.3f} {py:.3f} {WALL_HEIGHT / 2:.3f}" '
        f'size="{sx:.3f} {sy:.3f} {WALL_HEIGHT / 2:.3f}" rgba="0.30 0.30 0.34 1"/>'
        for n, (px, py, sx, sy) in enumerate(
            [
                (half, 0.0, half, WALL_THICKNESS),
                (half, size, half, WALL_THICKNESS),
                (0.0, half, WALL_THICKNESS, half),
                (size, half, WALL_THICKNESS, half),
            ]
        )
    )

    markers = ""
    if include_visual_markers:
        # Path markers are repositioned from Python each frame; 60 covers the
        # visible lookahead without bloating the model.
        path_sites = "\n".join(
            f'    <site name="path_{i}" pos="0 0 -5" size="0.07" rgba="0.20 0.70 1.00 0.85"/>'
            for i in range(60)
        )
        markers = f"""
    <site name="goal_marker" pos="0 0 -5" size="0.45" rgba="0.15 0.90 0.35 0.55"/>
{path_sites}"""

    quat_w = float(np.cos(start_heading / 2.0))
    quat_z = float(np.sin(start_heading / 2.0))

    return f"""
<mujoco model="pcnav_arena">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>

  <default>
    <geom condim="4" friction="1.0 0.01 0.001"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.22 0.24 0.26"
             rgb2="0.28 0.30 0.33" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="{int(size)} {int(size)}"
              reflectance="0.05"/>
  </asset>

  <worldbody>
    <light name="sun" pos="{half} {half} 12" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" pos="{half} {half} 0" size="{half} {half} 0.1"
          material="grid_mat"/>

{walls}
{obstacles}
{markers}

    <body name="chassis" pos="{start_xy[0]:.3f} {start_xy[1]:.3f} {CHASSIS_RIDE_HEIGHT:.3f}"
          quat="{quat_w:.6f} 0 0 {quat_z:.6f}">
      <freejoint name="chassis_free"/>
      <geom name="chassis_geom" type="box"
            size="{CHASSIS_HALF_LENGTH} {CHASSIS_HALF_WIDTH} {CHASSIS_HALF_HEIGHT}"
            mass="{CHASSIS_MASS}" rgba="0.85 0.55 0.15 1"/>
      <site name="robot_origin" pos="0 0 0" size="0.05" rgba="1 0 0 0.4"/>
{_wheel_body("wheel_fl", WHEEL_BASE, WHEEL_TRACK)}
{_wheel_body("wheel_fr", WHEEL_BASE, -WHEEL_TRACK)}
{_wheel_body("wheel_rl", -WHEEL_BASE, WHEEL_TRACK)}
{_wheel_body("wheel_rr", -WHEEL_BASE, -WHEEL_TRACK)}
    </body>
  </worldbody>

  <actuator>
    <velocity name="drive_fl" joint="wheel_fl_joint" kv="40" ctrlrange="-30 30"/>
    <velocity name="drive_fr" joint="wheel_fr_joint" kv="40" ctrlrange="-30 30"/>
    <velocity name="drive_rl" joint="wheel_rl_joint" kv="40" ctrlrange="-30 30"/>
    <velocity name="drive_rr" joint="wheel_rr_joint" kv="40" ctrlrange="-30 30"/>
  </actuator>
</mujoco>
""".strip()


def body_velocity_to_wheel_speeds(
    forward_speed: float, yaw_rate: float
) -> tuple[float, float, float, float]:
    """Skid-steer inverse kinematics: (v, omega) -> per-wheel angular speeds.

    This is the frozen low-level controller. The navigation policy never sees it,
    exactly as the paper's policy never sees the locomotion controller.

    Returns speeds ordered (front-left, front-right, rear-left, rear-right).
    """
    left_speed = (forward_speed - yaw_rate * WHEEL_TRACK) / WHEEL_RADIUS
    right_speed = (forward_speed + yaw_rate * WHEEL_TRACK) / WHEEL_RADIUS
    return left_speed, right_speed, left_speed, right_speed


def robot_footprint_radius() -> float:
    """Effective planning radius of the chassis, kept consistent with the maps."""
    return max(ROBOT_RADIUS_M, float(np.hypot(CHASSIS_HALF_LENGTH, CHASSIS_HALF_WIDTH)))
