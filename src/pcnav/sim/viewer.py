"""Interactive MuJoCo visualization of a trained policy.

Renders one episode at a time in the passive MuJoCo viewer, with the reference
path drawn as a ribbon of site markers and the goal as a translucent sphere.
Seeing the *reference path* alongside the robot is the whole point: it is how you
tell at a glance whether the policy is following guidance, shortcutting past it,
or correctly ignoring a path that leads somewhere wrong.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..config import MAX_EPISODE_STEPS, NAV_POLICY_HZ, PATH_VERTEX_SPACING_M
from ..envs.mujoco_env import MuJoCoNavEnv

NUM_PATH_MARKERS = 60
MARKER_STRIDE_M = 0.5   # spacing between drawn path markers
HIDDEN_MARKER_Z = -5.0  # parked below the floor when unused


class PolicyViewer:
    """Drive a policy in MuJoCo and render it in the passive viewer."""

    def __init__(self, env: MuJoCoNavEnv, policy, env_id: int = 0):
        if env.num_envs <= env_id:
            raise ValueError(f"env_id {env_id} out of range for {env.num_envs} envs")
        self.env = env
        self.policy = policy
        self.env_id = env_id
        self.mujoco = env.mujoco

    # ------------------------------------------------------------- markers

    def _marker_ids(self, model) -> tuple[int, list[int]]:
        goal_id = self.mujoco.mj_name2id(model, self.mujoco.mjtObj.mjOBJ_SITE, "goal_marker")
        path_ids = [
            self.mujoco.mj_name2id(model, self.mujoco.mjtObj.mjOBJ_SITE, f"path_{i}")
            for i in range(NUM_PATH_MARKERS)
        ]
        return goal_id, [i for i in path_ids if i >= 0]

    def _update_markers(self, model, data) -> None:
        """Point the goal and path markers at the current episode's geometry."""
        goal_id, path_ids = self._marker_ids(model)
        env_id = self.env_id

        if goal_id >= 0:
            goal = self.env.goal_position[env_id].numpy()
            model.site_pos[goal_id] = [goal[0], goal[1], 0.5]

        path = self.env.reference_path[env_id].numpy()
        path_len = int(self.env.reference_path_len[env_id])

        # Spread the fixed marker budget over the whole route rather than the first
        # N vertices. Routes here run 60-100 m; a fixed 0.5 m stride would draw only
        # the first 30 m and make a wrong-goal path look identical to a correct one.
        if path_len > len(path_ids):
            stride = max(1, path_len // len(path_ids))
        else:
            stride = max(1, int(round(MARKER_STRIDE_M / PATH_VERTEX_SPACING_M)))

        for slot, site_id in enumerate(path_ids):
            vertex = slot * stride
            if path_len > 0 and vertex < path_len:
                model.site_pos[site_id] = [path[vertex, 0], path[vertex, 1], 0.15]
            else:
                # No path (or past its end): park the marker out of sight.
                model.site_pos[site_id] = [0.0, 0.0, HIDDEN_MARKER_Z]

    # ---------------------------------------------------------------- loop

    def run(self, max_episodes: int = 10, realtime: bool = True) -> None:
        """Open the viewer and run episodes until the window is closed."""
        import mujoco.viewer

        observation = self.env.observe()
        model = self.env.models[self.env_id]
        data = self.env.datas[self.env_id]
        self._update_markers(model, data)

        step_period = 1.0 / NAV_POLICY_HZ
        episodes = 0

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 22.0
            viewer.cam.elevation = -40.0

            while viewer.is_running() and episodes < max_episodes:
                frame_start = time.time()

                with torch.no_grad():
                    action = self.policy.act_deterministic(observation)
                observation, reward, done, info = self.env.step(action)
                self.policy.reset_hidden(done)

                # A reset swaps in a freshly compiled model, so the viewer must be
                # rebound; MuJoCo's passive viewer cannot swap models in place.
                if bool(done[self.env_id]):
                    episodes += 1
                    outcome = (
                        "SUCCESS" if bool(info["success"][self.env_id])
                        else "TIPPED" if bool(info["tipped_over"][self.env_id])
                        else "COLLISION" if bool(info["collision"][self.env_id])
                        else "TIMEOUT"
                    )
                    print(f"episode {episodes}: {outcome}")
                    break

                self._update_markers(model, data)
                viewer.sync()

                if realtime:
                    remaining = step_period - (time.time() - frame_start)
                    if remaining > 0:
                        time.sleep(remaining)

        return None


def render_episode_frames(
    env: MuJoCoNavEnv,
    policy,
    env_id: int = 0,
    width: int = 960,
    height: int = 640,
    max_steps: int = MAX_EPISODE_STEPS,
    camera_distance: float = 14.0,
    camera_elevation: float = -55.0,
) -> tuple[list[np.ndarray], str]:
    """Offscreen render of one episode, for saving video without a display.

    The camera tracks the robot from above and behind. A fixed world camera is
    nearly useless on a 30 m arena -- the robot ends up a few pixels across, and
    the behaviour you are trying to diagnose becomes invisible.

    Returns (frames, outcome).
    """
    mujoco = env.mujoco
    model = env.models[env_id]
    data = env.datas[env_id]
    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = camera_distance
    camera.elevation = camera_elevation

    frames: list[np.ndarray] = []
    observation = env.observe()
    outcome = "timeout"

    for _ in range(max_steps):
        with torch.no_grad():
            action = policy.act_deterministic(observation)
        observation, _, done, info = env.step(action)
        policy.reset_hidden(done)

        camera.lookat[:] = [float(env.position[env_id, 0]), float(env.position[env_id, 1]), 0.0]
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

        if bool(done[env_id]):
            if bool(info["success"][env_id]):
                outcome = "success"
            elif bool(info["tipped_over"][env_id]):
                outcome = "tipped"
            elif bool(info["collision"][env_id]):
                outcome = "collision"
            break

    renderer.close()
    return frames, outcome
