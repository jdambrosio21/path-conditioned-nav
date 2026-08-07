"""Asymmetric actor-critic network.

The actor is restricted to what a real robot could sense. The critic receives
that **plus** privileged simulator state and the true optimal path.

The word that matters is *plus*. An asymmetric critic must observe a strict
**superset** of the actor -- extra information, never substituted information.
Handing the critic the optimal path *instead of* the path the actor actually
received breaks that, and the failure is subtle: for one underlying state the
actor's expected return depends heavily on which reference path it was given,
but a critic that cannot tell those cases apart predicts a single blended value.
The resulting advantages are biased per condition, and whichever condition the
policy happens to do best in gets systematically positive advantage, reinforcing
itself while the others are suppressed.

That is not hypothetical. Trained on a mixture, this codebase reached 0.77
success on NONE (the one condition the critic could identify, via the has_path
observation flag) while every path-bearing condition sat near 0.00 -- despite
each of them learning to >0.95 when trained in isolation.

So the critic encodes both paths: the one the actor saw, and the truth. The
encoder is shared between them, since they are the same modality; the two
contexts are distinguished by position in the concatenation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from ..config import ACTION_DIM, OBS_DIM, PRIV_DIM, PolicyConfig
from .path_encoder import TemporallyConsistentDropout, WaypointEncoder


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
    """Gaussian policy plus asymmetric value function."""

    def __init__(self, config: PolicyConfig, num_envs: int):
        super().__init__()
        self.config = config
        dim = config.embed_dim

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
        # state + context(actor's path) + context(true optimal path)
        self.critic_trunk = _build_mlp(3 * dim, config.trunk_hidden, 1)

        # Applied to the actor's proprioceptive/exteroceptive vector only.
        self.obs_dropout = TemporallyConsistentDropout(
            num_envs, OBS_DIM, config.dropout if config.temporally_consistent_dropout else 0.0
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ heads

    def action_mean(
        self, obs: torch.Tensor, path: torch.Tensor, dropout_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Deterministic action, from proprioception + the observed reference path."""
        obs = self.obs_dropout(obs, dropout_mask)
        state = self.actor_state_encoder(obs)
        context = self.actor_path_encoder(path, state)
        return self.actor_trunk(torch.cat([state, context], dim=-1))

    def value(
        self,
        obs: torch.Tensor,
        priv: torch.Tensor,
        optimal_path: torch.Tensor,
        observed_path: torch.Tensor,
    ) -> torch.Tensor:
        """Value estimate from privileged state, the actor's path, and the truth.

        `observed_path` is what the actor was given -- possibly corrupted, possibly
        absent. Without it the critic cannot distinguish an episode where the actor
        is following good guidance from one where it is being actively misled, and
        must average over both.
        """
        state = self.critic_state_encoder(torch.cat([obs, priv], dim=-1))
        observed_context = self.critic_path_encoder(observed_path, state)
        optimal_context = self.critic_path_encoder(optimal_path, state)
        return self.critic_trunk(
            torch.cat([state, observed_context, optimal_context], dim=-1)
        ).squeeze(-1)

    # ------------------------------------------------------------ distribution

    def distribution(
        self, obs: torch.Tensor, path: torch.Tensor, dropout_mask: torch.Tensor | None = None
    ) -> Normal:
        mean = self.action_mean(obs, path, dropout_mask)
        log_std = self.log_std.clamp(*self.log_std_bounds)
        return Normal(mean, log_std.exp().expand_as(mean))

    @torch.no_grad()
    def act(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Sample an action for rollout.

        The distribution parameters are returned alongside the sample because the
        PPO update needs the *collection-time* mean and std to compute an exact
        Gaussian KL. Estimating KL from importance ratios instead is heavy-tailed:
        a handful of tail samples dominate the mean and produce spurious spikes,
        which then drive the adaptive learning rate off a cliff.
        """
        dist = self.distribution(batch["obs"], batch["path"], batch.get("dropout_mask"))
        action = dist.sample()
        return {
            "action": action,
            "log_prob": dist.log_prob(action).sum(-1),
            "value": self.value(
                batch["obs"], batch["priv"], batch["opt_path"], batch["path"]
            ),
            "mean": dist.mean,
            "std": dist.stddev,
        }

    @torch.no_grad()
    def act_deterministic(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Mean action, for evaluation and visualization."""
        return self.action_mean(batch["obs"], batch["path"], batch.get("dropout_mask"))

    def evaluate(
        self, batch: dict[str, torch.Tensor], action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-score stored transitions during a PPO update.

        Returns (log_prob, entropy, value, mean, std).
        """
        dist = self.distribution(batch["obs"], batch["path"], batch.get("dropout_mask"))
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value(batch["obs"], batch["priv"], batch["opt_path"], batch["path"])
        return log_prob, entropy, value, dist.mean, dist.stddev
