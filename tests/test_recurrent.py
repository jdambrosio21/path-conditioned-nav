"""Tests for the Spatially-Enhanced Recurrent Unit and sequence replay.

The headline test is `test_sequence_replay_reproduces_rollout_exactly`. Recurrent
PPO's defining hazard is that the update re-scores stored actions through
*different* hidden states than produced them, which silently corrupts the
importance ratio and never raises. This is the third instance of that class of
bug in this codebase (after the dropout mask and the KL estimator), so it gets a
direct test rather than trust.
"""

from __future__ import annotations

import torch

from pcnav.config import NUM_WAYPOINTS, OBS_DIM, PRIV_DIM, PolicyConfig
from pcnav.models import PathConditionedActorCritic, RecurrentMemory, SpatiallyEnhancedGRUCell


def make_batch(batch_size: int = 8) -> dict[str, torch.Tensor]:
    waypoints = torch.randn(batch_size, NUM_WAYPOINTS, 3)
    waypoints[..., 2] = 1.0
    return {
        "obs": torch.randn(batch_size, OBS_DIM),
        "path": waypoints,
        "priv": torch.randn(batch_size, PRIV_DIM),
        "opt_path": waypoints.clone(),
    }


# --------------------------------------------------------------------------
# The cell itself
# --------------------------------------------------------------------------


def test_cell_output_shape_and_finiteness():
    cell = SpatiallyEnhancedGRUCell(12, 16)
    hidden = cell(torch.randn(5, 12), torch.zeros(5, 16))
    assert hidden.shape == (5, 16)
    assert torch.isfinite(hidden).all()


def test_spatial_term_starts_near_unity():
    """Initialized so the cell begins as a plain GRU and departs from it.

    A randomly scaled multiplicative term would either squash the candidate state
    to zero or saturate the tanh, stalling learning before it starts.
    """
    cell = SpatiallyEnhancedGRUCell(12, 16)
    spatial = cell.input_to_spatial(torch.randn(5, 12))
    assert torch.allclose(spatial, torch.ones_like(spatial), atol=1e-6)


def test_spatial_term_is_multiplicative_not_additive():
    """The defining property: the observation *scales* the candidate state.

    Scaling the spatial pathway must change the output even when the additive
    pathways are held fixed -- otherwise this is a GRU with extra parameters.
    """
    torch.manual_seed(0)
    cell = SpatiallyEnhancedGRUCell(12, 16)
    x, hidden = torch.randn(4, 12), torch.randn(4, 16)
    baseline = cell(x, hidden)

    with torch.no_grad():
        cell.input_to_spatial.bias.mul_(2.0)
    scaled = cell(x, hidden)
    assert not torch.allclose(baseline, scaled, atol=1e-5)


def test_hidden_state_carries_information_across_steps():
    """Two different histories must produce different states from identical input."""
    torch.manual_seed(0)
    memory = RecurrentMemory(12, 16, num_layers=1)
    x = torch.randn(3, 12)

    _, hidden_a = memory(torch.randn(3, 12), memory.initial_hidden(3, torch.device("cpu")))
    _, hidden_b = memory(torch.randn(3, 12), memory.initial_hidden(3, torch.device("cpu")))
    out_a, _ = memory(x, hidden_a)
    out_b, _ = memory(x, hidden_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


# --------------------------------------------------------------------------
# Sequence handling
# --------------------------------------------------------------------------


def test_sequence_matches_stepwise_iteration():
    torch.manual_seed(0)
    memory = RecurrentMemory(12, 16, num_layers=2)
    steps, envs = 6, 4
    x = torch.randn(steps, envs, 12)
    dones = torch.zeros(steps, envs)

    initial = memory.initial_hidden(envs, torch.device("cpu"))
    sequence = memory.forward_sequence(x, initial, dones)

    hidden = initial
    for t in range(steps):
        output, hidden = memory(x[t], hidden)
        assert torch.allclose(sequence[t], output, atol=1e-6)


def test_done_flag_clears_memory_for_the_next_step():
    """A finished episode must not leak its memory into the next one."""
    torch.manual_seed(0)
    memory = RecurrentMemory(12, 16, num_layers=1)
    steps, envs = 4, 3
    x = torch.randn(steps, envs, 12)

    dones = torch.zeros(steps, envs)
    dones[1, 0] = 1.0  # environment 0 terminates at step 1

    initial = memory.initial_hidden(envs, torch.device("cpu"))
    with_reset = memory.forward_sequence(x, initial, dones)
    without_reset = memory.forward_sequence(x, initial, torch.zeros_like(dones))

    # Steps up to and including the terminal one are unaffected...
    assert torch.allclose(with_reset[1, 0], without_reset[1, 0], atol=1e-6)
    # ...and everything after it diverges for that environment only.
    assert not torch.allclose(with_reset[2, 0], without_reset[2, 0], atol=1e-5)
    assert torch.allclose(with_reset[2, 1], without_reset[2, 1], atol=1e-6)


def test_reset_applies_after_the_terminal_action_not_before():
    """Ordering guard.

    The action at the terminal step was chosen *before* the environment ended, so
    its hidden state must still reflect the episode's history. Zeroing beforehand
    would attribute a fresh episode's memory to the previous episode's last action.
    """
    torch.manual_seed(1)
    memory = RecurrentMemory(8, 8, num_layers=1)
    x = torch.randn(3, 2, 8)
    dones = torch.zeros(3, 2)
    dones[0, 0] = 1.0

    initial = torch.randn(1, 2, 8)
    output = memory.forward_sequence(x, initial, dones)
    expected_first, _ = memory(x[0], initial)
    assert torch.allclose(output[0], expected_first, atol=1e-6)


# --------------------------------------------------------------------------
# The invariant that matters
# --------------------------------------------------------------------------


def test_sequence_replay_reproduces_rollout_exactly():
    """Update-time replay must reproduce collection-time actions bit-for-bit.

    PPO's importance ratio assumes the stored actions are re-scored under the same
    network that produced them. With recurrence that means replaying from the
    recorded initial hidden state and re-applying the same episode resets. If this
    drifts, nothing errors -- the ratio is simply wrong and learning quietly
    degrades.
    """
    torch.manual_seed(0)
    steps, envs = 5, 6
    policy = PathConditionedActorCritic(PolicyConfig(dropout=0.0), envs).eval()
    policy.reset_hidden()

    initial_actor = policy.actor_hidden.clone()
    initial_critic = policy.critic_hidden.clone()

    observations = {k: [] for k in ("obs", "path", "priv", "opt_path", "dropout_mask")}
    actions, means, dones = [], [], []

    for t in range(steps):
        batch = make_batch(envs)
        batch["dropout_mask"] = torch.ones(envs, OBS_DIM)
        out = policy.act(batch)

        for key in observations:
            observations[key].append(batch[key])
        actions.append(out["action"])
        means.append(out["mean"])

        done = torch.zeros(envs, dtype=torch.bool)
        if t == 2:
            done[1] = done[4] = True          # mid-rollout terminations
        policy.reset_hidden(done)
        dones.append(done.float())

    stacked = {k: torch.stack(v) for k, v in observations.items()}
    replay_means = policy.evaluate_sequence(
        stacked,
        torch.stack(actions),
        initial_actor,
        initial_critic,
        torch.stack(dones),
    )[3]

    expected = torch.stack(means).reshape(steps * envs, -1)
    assert torch.allclose(replay_means, expected, atol=1e-5), (
        "sequence replay diverged from the rollout: the PPO ratio would be invalid"
    )


def test_feedforward_mode_still_works():
    """Recurrence is optional; the non-recurrent path must remain intact."""
    policy = PathConditionedActorCritic(
        PolicyConfig(dropout=0.0, use_recurrence=False), 4
    ).eval()
    batch = make_batch(4)
    out = policy.act(batch)
    assert out["action"].shape == (4, 2)
    assert all(torch.isfinite(v).all() for v in out.values())
    policy.reset_hidden(torch.ones(4, dtype=torch.bool))  # must be a no-op
