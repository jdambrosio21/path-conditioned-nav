"""MJCF model generation: turn a procedural :class:`MapData` into a MuJoCo scene.

The robot is a **differential-drive** base: two driven wheels on a common lateral
axis, plus two low-friction caster points fore and aft.

It was originally a four-wheel skid-steer, which turned out to be the wrong model.
A rigid skid-steer can only rotate by scrubbing every wheel sideways against full
friction, and measured yaw tracking was 0.04 rad/s achieved against 1.0 commanded
-- the robot simply could not turn a corner, giving 0.00 success in a maze where
the idealized dynamics scored 0.98. Shrinking the chassis did not help; the
resistance is structural.

Differential drive turns freely because only the two driven wheels resist
rotation, and the casters carry load without opposing yaw. It is also the closer
analogue to the paper's platform: a B2W is a wheeled *quadruped* whose legs
reposition, so it is far more manoeuvrable than a rigid skid-steer.

What remains for the sim-to-sim comparison is genuine and unmodelled by the
training dynamics: wheel/ground contact, finite traction, body roll, tip-over,
and real collision response.

Obstacles become cylinders and the arena is walled, matching the geometry the
PyTorch backend ray-casts against analytically, so both backends solve the
*same* navigation problem under different physics.
"""

from __future__ import annotations

import numpy as np

from ..config import ROBOT_RADIUS_M
from ..maps import MapData

# --- chassis geometry (metres, kilograms) ---
#
# Sized so the *whole* robot, wheels included, fits inside ROBOT_RADIUS_M. Maps,
# planning and collision all model the robot as a disc of that radius; the earlier
# chassis had a 0.569 m envelope against a 0.350 m planning radius, so it was 63%
# larger than every other component believed. That is invisible until the robot can
# turn -- an under-actuated one crawls straight and never notices -- and then it
# sweeps a quarter-metre of unmodelled geometry into walls the planner called
# clear. Fixing the controller turned 38-71% collisions into 100%.
#
# `robot_footprint_radius()` recomputes the envelope and a test pins it against
# ROBOT_RADIUS_M so the two cannot drift apart again.
CHASSIS_HALF_LENGTH = 0.22
CHASSIS_HALF_WIDTH = 0.15
CHASSIS_HALF_HEIGHT = 0.08
CHASSIS_MASS = 20.0
WHEEL_RADIUS = 0.10
WHEEL_HALF_WIDTH = 0.04
WHEEL_MASS = 1.2
WHEEL_TRACK = 0.16      # lateral offset of each driven wheel from the centreline
WHEEL_BASE = 0.0        # driven wheels sit on the chassis centre, so yaw is free
CASTER_OFFSET = 0.17    # longitudinal offset of the fore/aft support casters
CASTER_RADIUS = 0.035
# Casters sit slightly proud of the ground so the driven wheels carry the load.
# Sharing it equally starved the wheels of normal force and they simply slipped:
# 0.22 m/s achieved against 1.0 commanded, on a flat plane with no obstacles.
CASTER_LIFT = 0.006

# Differential drive still under-turns: the driven wheels slip longitudinally and
# the casters drag at 0.17 m from the yaw axis. Measured tracking was 40-60% of
# the commanded yaw rate, consistently -- a systematic gain error, not noise. A
# real low-level controller would be calibrated against exactly this, so the
# inverse kinematics carries a measured feedforward term.
YAW_FEEDFORWARD = 2.0
CHASSIS_RIDE_HEIGHT = WHEEL_RADIUS

OBSTACLE_HEIGHT = 1.2
WALL_HEIGHT = 1.0
WALL_THICKNESS = 0.2
STRUCTURE_HEIGHT = 1.2   # trap walls: U-pockets, gapped barriers, dead ends


def _wheel_body(name: str, x: float, y: float) -> str:
    """One driven wheel: a hinge about the body-lateral axis with a velocity motor."""
    return f"""
      <body name="{name}" pos="{x:.3f} {y:.3f} 0">
        <joint name="{name}_joint" type="hinge" axis="0 1 0" damping="0.05"/>
        <geom name="{name}_geom" type="cylinder" size="{WHEEL_RADIUS} {WHEEL_HALF_WIDTH}"
              quat="0.707107 0.707107 0 0" mass="{WHEEL_MASS}"
              friction="1.6 0.005 0.0001" rgba="0.15 0.15 0.17 1"/>
      </body>"""


def _caster(name: str, x: float) -> str:
    """A near-frictionless support sphere.

    Casters carry load without resisting yaw. Modelling the swivel joint explicitly
    would add a degree of freedom with no bearing on navigation, so the swivel is
    approximated by making the contact slippery.
    """
    return f"""
      <geom name="{name}" type="sphere" size="{CASTER_RADIUS}"
            pos="{x:.3f} 0 {CASTER_RADIUS - WHEEL_RADIUS + CASTER_LIFT:.3f}" mass="0.2"
            friction="0.02 0.001 0.0001" rgba="0.25 0.25 0.28 1"/>"""


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

    # Trap structures are oriented boxes; MuJoCo takes orientation as a quaternion
    # about z, so yaw is converted here rather than stored twice.
    structures = "\n".join(
        f'      <geom name="structure_{i}" type="box" '
        f'pos="{cx:.3f} {cy:.3f} {STRUCTURE_HEIGHT / 2:.3f}" '
        f'size="{half_len:.3f} {half_thick:.3f} {STRUCTURE_HEIGHT / 2:.3f}" '
        f'quat="{np.cos(yaw / 2):.6f} 0 0 {np.sin(yaw / 2):.6f}" '
        f'rgba="0.38 0.40 0.46 1"/>'
        for i, (cx, cy, half_len, half_thick, yaw) in enumerate(map_data.walls)
        if half_len > 0
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

  <!-- The offscreen framebuffer defaults to 640x480; recording at a larger size
       fails at render time unless it is declared here. -->
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>

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
{structures}
{obstacles}
{markers}

    <body name="chassis" pos="{start_xy[0]:.3f} {start_xy[1]:.3f} {CHASSIS_RIDE_HEIGHT:.3f}"
          quat="{quat_w:.6f} 0 0 {quat_z:.6f}">
      <freejoint name="chassis_free"/>
      <geom name="chassis_geom" type="box"
            size="{CHASSIS_HALF_LENGTH} {CHASSIS_HALF_WIDTH} {CHASSIS_HALF_HEIGHT}"
            mass="{CHASSIS_MASS}" rgba="0.85 0.55 0.15 1"/>
      <site name="robot_origin" pos="0 0 0" size="0.05" rgba="1 0 0 0.4"/>
{_wheel_body("wheel_left", 0.0, WHEEL_TRACK)}
{_wheel_body("wheel_right", 0.0, -WHEEL_TRACK)}
{_caster("caster_front", CASTER_OFFSET)}
{_caster("caster_rear", -CASTER_OFFSET)}
    </body>
  </worldbody>

  <actuator>
    <velocity name="drive_left" joint="wheel_left_joint" kv="30" ctrlrange="-40 40"/>
    <velocity name="drive_right" joint="wheel_right_joint" kv="30" ctrlrange="-40 40"/>
  </actuator>
</mujoco>
""".strip()


def body_velocity_to_wheel_speeds(forward_speed: float, yaw_rate: float) -> tuple[float, float]:
    """Differential-drive inverse kinematics: (v, omega) -> wheel angular speeds.

    This is the frozen low-level controller. The navigation policy never sees it,
    exactly as the paper's policy never sees the locomotion controller.

    Returns (left, right).
    """
    differential = yaw_rate * WHEEL_TRACK * YAW_FEEDFORWARD
    left_speed = (forward_speed - differential) / WHEEL_RADIUS
    right_speed = (forward_speed + differential) / WHEEL_RADIUS
    return left_speed, right_speed


def robot_footprint_radius() -> float:
    """Radius of the smallest disc containing the whole robot, wheels included.

    Must not exceed ROBOT_RADIUS_M, which is what the occupancy grid, the roadmap
    and the collision term all assume.
    """
    chassis = float(np.hypot(CHASSIS_HALF_LENGTH, CHASSIS_HALF_WIDTH))
    wheels = float(np.hypot(WHEEL_RADIUS, WHEEL_TRACK + WHEEL_HALF_WIDTH))
    casters = CASTER_OFFSET + CASTER_RADIUS
    return max(chassis, wheels, casters)
