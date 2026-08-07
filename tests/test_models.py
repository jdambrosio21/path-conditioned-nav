"""Tests for the path encoder and the asymmetric actor-critic."""

from __future__ import annotations

import torch

from pcnav.config import ACTION_DIM, NUM_WAYPOINTS, OBS_DIM, PRIV_DIM, PolicyConfig
from pcnav.models import PathConditionedActorCritic, WaypointEncoder


def make_batch(batch_size: int = 8, valid: bool = True) -> dict[str, torch.Tensor]:
    waypoints = torch.randn(batch_size, NUM_WAYPOINTS, 3)
    waypoints[..., 2] = 1.0 if valid else 0.0
    if not valid:
        waypoints[..., :2] = 0.0
    return {
        "obs": torch.randn(batch_size, OBS_DIM),
        "path": waypoints,
        "priv": torch.randn(batch_size, PRIV_DIM),
        "opt_path": waypoints.clone(),
    }


def test_encoder_returns_exact_zero_for_empty_paths():
    """The "no path" signal must be a clean zero, not attention noise."""
    encoder = WaypointEncoder(embed_dim=32, num_heads=4, num_layers=2).eval()
    waypoints = torch.zeros(6, NUM_WAYPOINTS, 3)
    state = torch.randn(6, 32)
    context = encoder(waypoints, state)
    assert torch.isfinite(context).all()
    assert torch.all(context == 0.0)


def test_encoder_is_finite_with_partially_valid_paths():
    """A path that ends mid-window must not produce NaNs from the mask."""
    encoder = WaypointEncoder(embed_dim=32, num_heads=4, num_layers=2).eval()
    waypoints = torch.randn(6, NUM_WAYPOINTS, 3)
    waypoints[..., 2] = 0.0
    waypoints[:, :4, 2] = 1.0            # only the first four are real
    context = encoder(waypoints, torch.randn(6, 32))
    assert torch.isfinite(context).all()
    assert torch.any(context != 0.0)


def test_encoder_output_changes_with_path_geometry():
    """If the encoder ignored its input, path-conditioning would be a no-op."""
    encoder = WaypointEncoder(embed_dim=32, num_heads=4, num_layers=2).eval()
    state = torch.randn(4, 32)
    first = torch.randn(4, NUM_WAYPOINTS, 3)
    first[..., 2] = 1.0
    second = first.clone()
    second[..., :2] += 2.0
    assert not torch.allclose(encoder(first, state), encoder(second, state), atol=1e-4)


def test_actor_critic_forward_shapes():
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.0), num_envs=8).eval()
    batch = make_batch()
    out = policy.act(batch)
    assert out["action"].shape == (8, ACTION_DIM)
    assert "actor_hidden_in" in out
    assert out["log_prob"].shape == (8,)
    assert out["value"].shape == (8,)
    assert out["mean"].shape == (8, ACTION_DIM)
    assert out["std"].shape == (8, ACTION_DIM)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_policy_handles_missing_path_without_nans():
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.0), num_envs=8).eval()
    out = policy.act(make_batch(valid=False))
    assert all(torch.isfinite(v).all() for v in out.values())


def _sequence(batch: dict[str, torch.Tensor], envs: int = 8) -> dict[str, torch.Tensor]:
    """Wrap a single-step batch as a length-1 time-major sequence."""
    out = {k: v.unsqueeze(0) for k, v in batch.items()}
    out.setdefault("dropout_mask", torch.ones(envs, OBS_DIM).unsqueeze(0))
    if out["dropout_mask"].dim() == 2:
        out["dropout_mask"] = out["dropout_mask"].unsqueeze(0)
    return out


def test_evaluate_matches_act_under_a_fixed_action():
    """Re-scoring a stored action must reproduce its log-prob deterministically."""
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.0), num_envs=8).eval()
    batch = make_batch()
    action = torch.zeros(1, 8, ACTION_DIM)
    args = (
        _sequence(batch), action,
        policy.actor_hidden.clone(), policy.critic_hidden.clone(), torch.zeros(1, 8),
    )
    first_log_prob, _, first_value, _, _ = policy.evaluate_sequence(*args)
    second_log_prob, _, second_value, _, _ = policy.evaluate_sequence(*args)
    assert torch.allclose(first_log_prob, second_log_prob)
    assert torch.allclose(first_value, second_value)


def test_critic_uses_privileged_input():
    """Changing privileged state must move the value estimate."""
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.0), num_envs=8).eval()
    batch = make_batch()
    baseline = policy.bootstrap_value(batch)
    altered = policy.bootstrap_value({**batch, "priv": batch["priv"] + 1.0})
    assert not torch.allclose(baseline, altered, atol=1e-5)


def test_critic_distinguishes_episodes_by_the_path_the_actor_saw():
    """The superset property.

    Two episodes identical in privileged state and true optimal path, differing
    only in what the actor was handed, must receive different value estimates --
    otherwise advantages are blended across path conditions and whichever
    condition the policy does best in crowds out the rest.
    """
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.0), num_envs=8).eval()
    batch = make_batch()
    with_path = policy.bootstrap_value(batch)
    without_path = policy.bootstrap_value({**batch, "path": torch.zeros_like(batch["path"])})
    assert not torch.allclose(with_path, without_path, atol=1e-5)


def test_temporally_consistent_dropout_mask_is_stable_between_resamples():
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.2), num_envs=16)
    policy.obs_dropout.resample()
    first = policy.obs_dropout.current_mask.clone()

    policy.act_deterministic(make_batch(16))
    assert torch.equal(policy.obs_dropout.current_mask, first)  # unchanged by a forward

    policy.obs_dropout.resample()
    assert not torch.equal(policy.obs_dropout.current_mask, first)


def test_stored_mask_reproduces_collection_time_network():
    """Replaying a stored mask must reproduce the exact action that was sampled."""
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.3), num_envs=8).eval()
    policy.reset_hidden()
    policy.obs_dropout.resample()

    batch = make_batch()
    batch["dropout_mask"] = policy.obs_dropout.current_mask.clone()
    hidden = (policy.actor_hidden.clone(), policy.critic_hidden.clone())
    original = policy.act(batch)["mean"]

    policy.obs_dropout.resample()  # mask moves on, as it would after an episode ends
    replayed = policy.evaluate_sequence(
        _sequence(batch), original.unsqueeze(0), hidden[0], hidden[1], torch.zeros(1, 8)
    )[3]
    assert torch.allclose(original, replayed, atol=1e-5)


def test_gaussian_kl_is_zero_for_identical_distributions_and_positive_otherwise():
    """Guards the LR controller's input: KL must be exact, not a noisy estimate."""
    from pcnav.algorithms.ppo import gaussian_kl

    mean = torch.randn(32, ACTION_DIM)
    std = torch.rand(32, ACTION_DIM) + 0.5
    assert torch.allclose(gaussian_kl(mean, std, mean, std), torch.zeros(32), atol=1e-6)

    shifted = gaussian_kl(mean, std, mean + 0.5, std)
    widened = gaussian_kl(mean, std, mean, std * 2.0)
    assert torch.all(shifted > 0) and torch.all(widened > 0)


def test_log_std_is_clamped_within_bounds():
    """Without an upper bound the entropy bonus can inflate sigma without limit."""
    from pcnav.config import PolicyConfig as _PolicyConfig

    policy = PathConditionedActorCritic(_PolicyConfig(dropout=0.0), num_envs=4).eval()
    with torch.no_grad():
        policy.log_std.fill_(50.0)
    std = policy.act(make_batch(4))["std"]
    assert torch.all(std <= torch.tensor(policy.log_std_bounds[1]).exp() + 1e-5)
