"""Recurrent, asymmetric actor-critic network.

Two structural commitments, both taken from the papers.

**Asymmetry.** The actor is restricted to what a real robot could sense. The
critic receives that **plus** privileged simulator state and the true optimal
path. The word that matters is *plus* — an asymmetric critic must observe a
strict **superset** of the actor, never substituted information. Handing the
critic the optimal path *instead of* the path the actor received breaks that, and
the failure is subtle: for one underlying state the actor's expected return
depends heavily on which reference path it was given, but a critic that cannot
tell those cases apart predicts a single blended value. Advantages then become
biased per condition, and whichever condition the policy happens to do best in
gets systematically positive advantage and crowds out the rest.

That is not hypothetical. Trained on a mixture, this codebase reached 0.77
success on NONE (the one condition the critic could identify, via the has_path
observation flag) while every path-bearing condition sat near 0.00 -- despite
each of them learning to >0.95 when trained in isolation.

**Where memory sits.** Proprioception and exteroception feed *into* the recurrent
unit; the path embedding is concatenated *after* it. That is Haro et al.'s
arrangement, and their reasoning is worth keeping:

    "Proprioceptive features are integrated upstream of the SRU and contribute to
    its hidden state. This design reflects the assumption that the reference path
    remains fixed during an episode and therefore does not require temporal
    integration within the recurrent memory."

The path does not change during an episode, so pushing it through recurrence
would spend capacity integrating a constant.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from ..config import ACTION_DIM, OBS_DIM, PRIV_DIM, PolicyConfig
from .path_encoder import TemporallyConsistentDropout, WaypointEncoder
from .recurrent import RecurrentMemory


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Standard ELU trunk, matching the rsl-rl convention the paper builds on."""
    layers: list[nn.Module] = []
    prev = input_dim
    for width in hidden_dims:
        layers += [nn.Linear(prev, width), nn.ELU()]
        prev = width
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class PathConditionedActorCritic(nn.Module):
    """Gaussian policy with spatially-enhanced recurrent memory, plus a privileged critic."""

    def __init__(self, config: PolicyConfig, num_envs: int):
        super().__init__()
        self.config = config
        self.num_envs = num_envs
        dim = config.embed_dim
        self.recurrent = config.use_recurrence

        # --- actor ---
        self.actor_state_encoder = nn.Sequential(
            nn.Linear(OBS_DIM, dim), nn.ELU(), nn.Linear(dim, dim)
        )
        self.actor_path_encoder = WaypointEncoder(dim, config.num_heads, config.encoder_layers)
        self.actor_trunk = _build_mlp(2 * dim, config.trunk_hidden, ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), config.init_log_std))
        # Bounds on the action noise. Without an upper bound the entropy bonus can
        # inflate sigma indefinitely when the advantage signal is weak, which both
        # destroys the policy and produces the tail samples that wreck KL estimates.
        self.log_std_bounds = config.log_std_bounds

        # --- critic (privileged) ---
        self.critic_state_encoder = nn.Sequential(
            nn.Linear(OBS_DIM + PRIV_DIM, dim), nn.ELU(), nn.Linear(dim, dim)
        )
        self.critic_path_encoder = WaypointEncoder(dim, config.num_heads, config.encoder_layers)
        self.critic_trunk = _build_mlp(3 * dim, config.trunk_hidden, 1)

        # --- memory ---
        if self.recurrent:
            self.actor_memory = RecurrentMemory(dim, dim, config.memory_layers)
            self.critic_memory = RecurrentMemory(dim, dim, config.memory_layers)
            self.register_buffer(
                "actor_hidden",
                self.actor_memory.initial_hidden(num_envs, torch.device("cpu")),
                persistent=False,
            )
            self.register_buffer(
                "critic_hidden",
                self.critic_memory.initial_hidden(num_envs, torch.device("cpu")),
                persistent=False,
            )

        self.obs_dropout = TemporallyConsistentDropout(
            num_envs, OBS_DIM, config.dropout if config.temporally_consistent_dropout else 0.0
        )

    @property
    def hidden_size(self) -> int:
        return self.config.embed_dim if self.recurrent else 1

    @property
    def memory_layers(self) -> int:
        return self.config.memory_layers if self.recurrent else 1

    # -------------------------------------------------------------- hidden state

    @torch.no_grad()
    def reset_hidden(self, done: torch.Tensor | None = None) -> None:
        """Zero the memory for finished episodes, or all of them if `done` is None."""
        if not self.recurrent:
            return
        if done is None:
            self.actor_hidden.zero_()
            self.critic_hidden.zero_()
            return
        keep = (~done).float().view(1, -1, 1)
        self.actor_hidden.mul_(keep)
        self.critic_hidden.mul_(keep)

    # ---------------------------------------------------------------- assembling

    def _actor_head(
        self, obs: torch.Tensor, path: torch.Tensor, memory_output: torch.Tensor
    ) -> torch.Tensor:
        context = self.actor_path_encoder(path, memory_output)
        return self.actor_trunk(torch.cat([memory_output, context], dim=-1))

    def _critic_head(
        self,
        optimal_path: torch.Tensor,
        observed_path: torch.Tensor,
        memory_output: torch.Tensor,
    ) -> torch.Tensor:
        observed_context = self.critic_path_encoder(observed_path, memory_output)
        optimal_context = self.critic_path_encoder(optimal_path, memory_output)
        return self.critic_trunk(
            torch.cat([memory_output, observed_context, optimal_context], dim=-1)
        ).squeeze(-1)

    def _distribution(self, mean: torch.Tensor) -> Normal:
        std = self.log_std.clamp(*self.log_std_bounds).exp().expand_as(mean)
        return Normal(mean, std)

    # ------------------------------------------------------------------- rollout

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Sample an action and advance the internal memory by one step.

        The hidden state *entering* this step is returned so the rollout buffer can
        record it. The PPO update replays each sequence from the recorded state,
        which is the only way the re-scored actions come from the same network that
        produced them -- the same invariant the dropout mask already needed.
        """
        obs = self.obs_dropout(batch["obs"], batch.get("dropout_mask"))
        actor_state = self.actor_state_encoder(obs)
        critic_state = self.critic_state_encoder(torch.cat([batch["obs"], batch["priv"]], dim=-1))

        if self.recurrent:
            actor_hidden_in = self.actor_hidden.clone()
            critic_hidden_in = self.critic_hidden.clone()
            actor_out, new_actor = self.actor_memory(actor_state, actor_hidden_in)
            critic_out, new_critic = self.critic_memory(critic_state, critic_hidden_in)
            self.actor_hidden.copy_(new_actor)
            self.critic_hidden.copy_(new_critic)
        else:
            zeros = torch.zeros(1, obs.shape[0], 1, device=obs.device)
            actor_hidden_in = critic_hidden_in = zeros
            actor_out, critic_out = actor_state, critic_state

        mean = self._actor_head(obs, batch["path"], actor_out)
        distribution = self._distribution(mean)
        action = distribution.sample()

        return {
            "action": action,
            "log_prob": distribution.log_prob(action).sum(-1),
            "value": self._critic_head(batch["opt_path"], batch["path"], critic_out),
            "mean": mean,
            "std": distribution.stddev,
            "actor_hidden_in": actor_hidden_in,
            "critic_hidden_in": critic_hidden_in,
        }

    @torch.no_grad()
    def act_deterministic(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean action, advancing memory. For evaluation and visualization."""
        obs = self.obs_dropout(batch["obs"], batch.get("dropout_mask"))
        actor_state = self.actor_state_encoder(obs)
        if self.recurrent:
            actor_out, new_hidden = self.actor_memory(actor_state, self.actor_hidden)
            self.actor_hidden.copy_(new_hidden)
        else:
            actor_out = actor_state
        return self._actor_head(obs, batch["path"], actor_out)

    @torch.no_grad()
    def bootstrap_value(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Value at the end of a rollout. Does not advance memory."""
        critic_state = self.critic_state_encoder(torch.cat([batch["obs"], batch["priv"]], dim=-1))
        if self.recurrent:
            critic_out, _ = self.critic_memory(critic_state, self.critic_hidden)
        else:
            critic_out = critic_state
        return self._critic_head(batch["opt_path"], batch["path"], critic_out)

    # -------------------------------------------------------------------- update

    def evaluate_sequence(
        self,
        batch: dict[str, torch.Tensor],
        actions: torch.Tensor,
        actor_hidden: torch.Tensor,
        critic_hidden: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score a rollout segment, replaying memory from its recorded start.

        All observation tensors are time-major, shape (T, B, ...). Returns
        (log_prob, entropy, value, mean, std) flattened to (T*B, ...).
        """
        timesteps, envs = actions.shape[0], actions.shape[1]

        def flatten(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(timesteps * envs, *tensor.shape[2:])

        obs = flatten(batch["obs"])
        obs_dropped = self.obs_dropout(obs, flatten(batch["dropout_mask"]))
        actor_state = self.actor_state_encoder(obs_dropped)
        critic_state = self.critic_state_encoder(
            torch.cat([obs, flatten(batch["priv"])], dim=-1)
        )

        if self.recurrent:
            actor_out = self.actor_memory.forward_sequence(
                actor_state.view(timesteps, envs, -1), actor_hidden, dones
            ).reshape(timesteps * envs, -1)
            critic_out = self.critic_memory.forward_sequence(
                critic_state.view(timesteps, envs, -1), critic_hidden, dones
            ).reshape(timesteps * envs, -1)
        else:
            actor_out, critic_out = actor_state, critic_state

        mean = self._actor_head(obs_dropped, flatten(batch["path"]), actor_out)
        distribution = self._distribution(mean)
        value = self._critic_head(
            flatten(batch["opt_path"]), flatten(batch["path"]), critic_out
        )
        flat_actions = actions.reshape(timesteps * envs, -1)

        return (
            distribution.log_prob(flat_actions).sum(-1),
            distribution.entropy().sum(-1),
            value,
            mean,
            distribution.stddev,
        )
