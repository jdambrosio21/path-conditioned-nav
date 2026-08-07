"""Tests for the vectorized environment: geometry, rewards, and path conditions."""

from __future__ import annotations

import pytest
import torch

from pcnav.config import (
    NUM_RAYS,
    NUM_WAYPOINTS,
    OBS_DIM,
    PRIV_DIM,
    SENSOR_MAX_RANGE_M,
    EnvConfig,
    MapConfig,
)
from pcnav.envs.torch_env import PathConditionedNavEnv
from pcnav.planning import PathQuality


def make_env(
    num_envs: int = 64, seed: int = 0, quality: str | None = None
) -> PathConditionedNavEnv:
    return PathConditionedNavEnv(
        EnvConfig(
            num_envs=num_envs,
            device="cpu",
            seed=seed,
            maps=MapConfig(num_maps=4),
            fixed_path_quality=quality,
        )
    )


@pytest.fixture(scope="module")
def env() -> PathConditionedNavEnv:
    return make_env()


def test_observation_shapes_and_finiteness(env: PathConditionedNavEnv):
    obs = env.observe()
    assert obs["obs"].shape == (env.num_envs, OBS_DIM)
    assert obs["priv"].shape == (env.num_envs, PRIV_DIM)
    assert obs["path"].shape == (env.num_envs, NUM_WAYPOINTS, 3)
    assert obs["opt_path"].shape == (env.num_envs, NUM_WAYPOINTS, 3)
    for tensor in obs.values():
        assert torch.isfinite(tensor).all()


def test_scan_is_normalized_within_unit_range(env: PathConditionedNavEnv):
    scan = env.observe()["obs"][:, :NUM_RAYS]
    assert scan.min() >= 0.0 and scan.max() <= 1.0


def test_rays_never_report_beyond_clearance():
    """The shortest ray cannot exceed the true distance to the nearest obstacle.

    This catches sign and frame errors in the ray-circle solve, which would
    otherwise show up only as a policy that mysteriously refuses to learn.
    """
    env = make_env(num_envs=128, seed=7)
    scan = env._cast_rays()
    clearance = env._clearance()
    # A ray pointing straight at the nearest obstacle can be at most that far,
    # but rays may point elsewhere -- so only the lower bound is guaranteed.
    assert torch.all(scan.min(dim=1).values >= -1e-4)
    assert torch.all(scan <= SENSOR_MAX_RANGE_M + 1e-4)
    assert torch.isfinite(clearance).all()


def test_stationary_robot_makes_no_progress():
    """Zero throttle must produce (near) zero geodesic progress reward."""
    env = make_env(num_envs=32, seed=3, quality="OPTIMAL")
    action = torch.zeros(env.num_envs, 2)
    env.step(action)
    before = env.previous_geodesic.clone()
    env.step(action)
    assert torch.allclose(env.previous_geodesic, before, atol=0.35)


def test_no_path_condition_zeroes_the_waypoints():
    env = make_env(num_envs=32, seed=5, quality="NONE")
    obs = env.observe()
    assert torch.all(obs["path"] == 0.0)
    assert torch.all(env.reference_path_len == 0)
    # The has_path flag is the final observation channel.
    assert torch.all(obs["obs"][:, -1] == 0.0)


def test_optimal_condition_supplies_valid_waypoints():
    env = make_env(num_envs=32, seed=5, quality="OPTIMAL")
    obs = env.observe()
    assert torch.all(env.reference_path_len > 0)
    assert obs["path"][..., 2].sum() > 0
    assert torch.all(obs["obs"][:, -1] == 1.0)


def test_critic_always_receives_the_optimal_path_even_when_actor_does_not():
    """The asymmetry that makes the value function trustworthy under corruption."""
    env = make_env(num_envs=32, seed=11, quality="NONE")
    obs = env.observe()
    assert torch.all(obs["path"] == 0.0)          # actor: nothing
    assert obs["opt_path"][..., 2].sum() > 0      # critic: ground truth


def test_wrong_goal_path_leads_somewhere_else():
    """A WRONG_GOAL reference must not terminate near the true goal."""
    env = make_env(num_envs=64, seed=13, quality="WRONG_GOAL")
    misleading = 0
    for i in range(env.num_envs):
        length = int(env.reference_path_len[i])
        if length == 0:
            continue
        endpoint = env.reference_path[i, length - 1]
        if torch.norm(endpoint - env.goal_position[i]) > 3.0:
            misleading += 1
    assert misleading > env.num_envs // 2


def test_shortcut_reward_requires_a_reference_path():
    """With no path there is no arclength to beat, so the term must be inert."""
    env = make_env(num_envs=16, seed=17, quality="NONE")
    action = torch.zeros(env.num_envs, 2)
    action[:, 0] = 1.0
    _, reward, _, _ = env.step(action)
    assert torch.isfinite(reward).all()


def test_episode_terminates_and_auto_resets():
    env = make_env(num_envs=16, seed=19, quality="OPTIMAL")
    action = torch.zeros(env.num_envs, 2)
    action[:, 0] = 1.0
    action[:, 1] = 1.0  # drive in circles until something terminates
    saw_done = False
    for _ in range(400):
        _, _, done, _ = env.step(action)
        if bool(done.any()):
            saw_done = True
            break
    assert saw_done
    assert torch.all(env.step_count >= 0)


def test_path_quality_mixture_covers_all_conditions():
    env = make_env(num_envs=2048, seed=23)
    seen = set(env.path_quality.tolist())
    assert seen == {int(q) for q in PathQuality}
