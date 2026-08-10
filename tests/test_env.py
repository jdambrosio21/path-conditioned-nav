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
    _, reward, _, info = env.step(action)
    assert torch.isfinite(reward).all()
    assert torch.all(info["reward_terms"]["shortcut"] == 0.0)


def test_shortcut_reward_never_pays_for_retreating():
    """Regression guard for a reward-hacking exploit.

    Backing away from the goal shortens the reference-path arclength faster than
    it costs geodesic distance, because a path is always longer than the geodesic.
    An unguarded `(d_geo - d_s).clamp(min=0)` therefore pays out for retreating,
    and the policy learns to drive at full reverse whenever a path is present.
    """
    env = make_env(num_envs=64, seed=29, quality="OPTIMAL")
    weights = env.config.reward
    reverse = torch.zeros(env.num_envs, 2)
    reverse[:, 0] = -1.0
    for _ in range(15):
        _, _, _, info = env.step(reverse)
        progress = info["reward_terms"]["progress"] / weights.progress
        bonus = info["reward_terms"]["shortcut"] / weights.shortcut
        # The invariant is conditional: some environments start facing sideways to
        # the goal, so reversing can still close distance. What must never happen
        # is a payout while losing ground.
        assert torch.all(bonus[progress <= 0.0] <= 1e-6)


def test_shortcut_reward_cannot_be_farmed_by_oscillating():
    """The core anti-exploit property.

    A policy that drives forward and back repeatedly must collect the shortcut
    bonus at most once, not once per cycle. A per-step one-sided bonus fails this;
    a running maximum passes it.
    """
    env = make_env(num_envs=32, seed=31, quality="SUBOPTIMAL")
    weights = env.config.reward
    # Forward and reverse limits differ 4x, so the actions are chosen to give equal
    # speed (0.5 m/s each way). Otherwise the "oscillation" ratchets steadily
    # forward and legitimately earns new credit every cycle.
    forward = torch.zeros(env.num_envs, 2)
    forward[:, 0] = 0.25
    backward = torch.zeros(env.num_envs, 2)
    backward[:, 0] = -1.0

    # An episode that terminates mid-test resets the bookkeeping, so only
    # environments that survive the whole sequence can be accounted.
    alive = torch.ones(env.num_envs, dtype=torch.bool)
    collected = torch.zeros(env.num_envs)
    first_cycle = torch.zeros(env.num_envs)
    for cycle in range(6):
        for action in (forward, backward):
            for _ in range(4):
                _, _, done, info = env.step(action)
                collected += (info["reward_terms"]["shortcut"] / weights.shortcut) * alive
                alive &= ~done
        if cycle == 0:
            first_cycle = collected.clone()

    assert bool(alive.any()), "no environment survived the oscillation test"
    # Later cycles revisit ground already credited, so they must add ~nothing.
    assert torch.all(collected[alive] <= first_cycle[alive] + 1e-3)


def test_shortcut_bonus_is_bounded_by_the_lead_achieved():
    """Total credit over an episode cannot exceed the lead over the path."""
    env = make_env(num_envs=64, seed=37, quality="SUBOPTIMAL")
    weights = env.config.reward
    forward = torch.zeros(env.num_envs, 2)
    forward[:, 0] = 1.0
    alive = torch.ones(env.num_envs, dtype=torch.bool)
    total = torch.zeros(env.num_envs)
    for _ in range(20):
        _, _, done, info = env.step(forward)
        total += (info["reward_terms"]["shortcut"] / weights.shortcut) * alive
        alive &= ~done
    assert bool(alive.any())
    assert torch.all(total[alive] <= env.best_lead[alive].clamp(min=0.0) + 1e-3)


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


def test_mujoco_robot_fits_inside_the_planning_radius():
    """Cross-component consistency: geometry vs the disc every planner assumes.

    Maps, roadmap and collision all model the robot as a disc of ROBOT_RADIUS_M.
    If the MuJoCo body is larger it cannot fit through gaps the planner considers
    clear, and the symptom is a wall of collisions with no obvious cause -- the
    earlier body had a 0.569 m envelope against a 0.350 m radius.
    """
    from pcnav.config import ROBOT_RADIUS_M
    from pcnav.sim.mjcf import robot_footprint_radius

    assert robot_footprint_radius() <= ROBOT_RADIUS_M, (
        f"MuJoCo robot ({robot_footprint_radius():.3f} m) exceeds the planning "
        f"radius ({ROBOT_RADIUS_M:.3f} m)"
    )
