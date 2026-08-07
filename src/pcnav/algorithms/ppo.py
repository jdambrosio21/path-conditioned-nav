"""Proximal Policy Optimization with a recurrent, asymmetric critic.

Structured after rsl-rl -- the library the paper uses -- including its adaptive
learning-rate schedule, which nudges the step size to hold the policy KL near a
target instead of following a fixed decay.

Recurrence changes the minibatching, and the change is not optional. Feedforward
PPO shuffles all (timestep, environment) transitions freely because each is
independent. With memory they are not: a transition's meaning depends on the
hidden state that preceded it, so a rollout must be replayed **in order, from a
recorded starting state**. Minibatches are therefore taken over *environments*,
each carrying its full time sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import PPOConfig
from ..models.actor_critic import PathConditionedActorCritic

# Observation tensors carried through the buffer, in the order the network wants.
OBSERVATION_KEYS = ("obs", "path", "priv", "opt_path", "dropout_mask")


def gaussian_kl(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
) -> torch.Tensor:
    """Exact KL( old || new ) for diagonal Gaussians, summed over action dims.

    Preferred over the ratio-based estimator for driving the learning-rate
    schedule. The ratio estimator is heavy-tailed -- a few samples drawn far into
    the tail of the old policy can dominate the batch mean and report a large KL
    even when `clip_fraction` shows the update was tiny. This closed form depends
    only on the distribution parameters, so it is bounded and noise-free.
    """
    variance_ratio = (old_std / new_std).pow(2)
    mean_term = ((new_mean - old_mean) / new_std).pow(2)
    return 0.5 * (variance_ratio + mean_term - 1.0 - variance_ratio.log()).sum(-1)


class RolloutBuffer:
    """On-policy storage, time-major as (steps, envs, ...) so sequences stay intact."""

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shapes: dict[str, tuple[int, ...]],
        action_dim: int,
        memory_layers: int,
        hidden_size: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.step = 0

        self.observations = {
            key: torch.zeros(num_steps, num_envs, *shape, device=device)
            for key, shape in obs_shapes.items()
        }
        self.actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        # Collection-time distribution parameters, for the exact KL computation.
        self.action_means = torch.zeros(num_steps, num_envs, action_dim, device=device)
        self.action_stds = torch.ones(num_steps, num_envs, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, device=device)
        self.advantages = torch.zeros(num_steps, num_envs, device=device)
        self.returns = torch.zeros(num_steps, num_envs, device=device)

        # Only the hidden state entering step 0 is stored. The update replays the
        # sequence forward from there, so intermediate states are recomputed under
        # the current parameters rather than reused stale.
        self.initial_actor_hidden = torch.zeros(
            memory_layers, num_envs, hidden_size, device=device
        )
        self.initial_critic_hidden = torch.zeros(
            memory_layers, num_envs, hidden_size, device=device
        )

    def start_rollout(self, actor_hidden: torch.Tensor, critic_hidden: torch.Tensor) -> None:
        self.step = 0
        self.initial_actor_hidden.copy_(actor_hidden)
        self.initial_critic_hidden.copy_(critic_hidden)

    def add(
        self,
        observation: dict[str, torch.Tensor],
        step_output: dict[str, torch.Tensor],
        reward: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        for key in self.observations:
            self.observations[key][self.step] = observation[key]
        self.actions[self.step] = step_output["action"]
        self.action_means[self.step] = step_output["mean"]
        self.action_stds[self.step] = step_output["std"]
        self.log_probs[self.step] = step_output["log_prob"]
        self.values[self.step] = step_output["value"]
        self.rewards[self.step] = reward
        self.dones[self.step] = done.float()
        self.step += 1

    def compute_returns(self, last_value: torch.Tensor, gamma: float, gae_lambda: float) -> None:
        """Generalized Advantage Estimation, walked backwards through the rollout."""
        running_advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            next_value = last_value if t == self.num_steps - 1 else self.values[t + 1]
            not_done = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * not_done - self.values[t]
            running_advantage = delta + gamma * gae_lambda * not_done * running_advantage
            self.advantages[t] = running_advantage
        self.returns = self.advantages + self.values

    def sequence_minibatches(self, num_minibatches: int):
        """Yield minibatches of whole environment sequences, in timestep order.

        Environments are shuffled between epochs but each keeps its full time
        sequence, because replaying memory requires contiguous ordered transitions.
        """
        # Normalize advantages once over the full rollout, so the scale is
        # consistent across minibatches rather than varying with each subset.
        advantages = self.advantages
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        permutation = torch.randperm(self.num_envs, device=self.device)
        size = max(1, self.num_envs // num_minibatches)
        action_dim = self.actions.shape[-1]

        for i in range(num_minibatches):
            env_ids = permutation[i * size : (i + 1) * size]
            if env_ids.numel() == 0:
                continue
            yield {
                "obs": {k: v[:, env_ids] for k, v in self.observations.items()},
                "actions": self.actions[:, env_ids],
                "old_means": self.action_means[:, env_ids].reshape(-1, action_dim),
                "old_stds": self.action_stds[:, env_ids].reshape(-1, action_dim),
                "old_log_probs": self.log_probs[:, env_ids].reshape(-1),
                "advantages": normalized[:, env_ids].reshape(-1),
                "returns": self.returns[:, env_ids].reshape(-1),
                "old_values": self.values[:, env_ids].reshape(-1),
                "actor_hidden": self.initial_actor_hidden[:, env_ids],
                "critic_hidden": self.initial_critic_hidden[:, env_ids],
                "dones": self.dones[:, env_ids],
            }


class PPO:
    """Clipped-surrogate PPO with value clipping and an adaptive KL-targeted LR."""

    def __init__(self, policy: PathConditionedActorCritic, config: PPOConfig, device: torch.device):
        self.policy = policy
        self.config = config
        self.device = device
        self.learning_rate = config.learning_rate
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        cfg = self.config
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "clip_frac": 0.0}
        num_updates = 0

        for _ in range(cfg.num_epochs):
            for batch in buffer.sequence_minibatches(cfg.num_minibatches):
                log_probs, entropy, values, means, stds = self.policy.evaluate_sequence(
                    batch["obs"],
                    batch["actions"],
                    batch["actor_hidden"],
                    batch["critic_hidden"],
                    batch["dones"],
                )

                advantages = batch["advantages"]
                old_log_probs = batch["old_log_probs"]

                # --- clipped policy surrogate ---
                ratio = (log_probs - old_log_probs).exp()
                unclipped = ratio * advantages
                clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # --- clipped value loss ---
                old_values, returns = batch["old_values"], batch["returns"]
                value_clipped = old_values + (values - old_values).clamp(
                    -cfg.clip_ratio, cfg.clip_ratio
                )
                value_loss = torch.max(
                    (values - returns).pow(2), (value_clipped - returns).pow(2)
                ).mean()

                entropy_bonus = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    - cfg.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = gaussian_kl(
                        batch["old_means"], batch["old_stds"], means, stds
                    ).mean()
                    clip_fraction = ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean()

                stats["policy_loss"] += float(policy_loss.detach())
                stats["value_loss"] += float(value_loss.detach())
                stats["entropy"] += float(entropy_bonus.detach())
                stats["kl"] += float(approx_kl)
                stats["clip_frac"] += float(clip_fraction)
                num_updates += 1

        for key in stats:
            stats[key] /= max(num_updates, 1)

        # Adapt once per iteration on the mean KL. Adapting per minibatch compounds
        # ~20 multiplicative steps per iteration and drives the learning rate into
        # its floor within a handful of iterations, stalling the run.
        if cfg.adaptive_lr:
            self._adapt_learning_rate(stats["kl"])

        with torch.no_grad():
            bounded_log_std = self.policy.log_std.clamp(*self.policy.log_std_bounds)
            stats["action_std"] = float(bounded_log_std.exp().mean())
        stats["learning_rate"] = self.learning_rate
        return stats

    def _adapt_learning_rate(self, approx_kl: float) -> None:
        """rsl-rl's schedule: shrink on KL overshoot, grow on undershoot.

        The step is deliberately gentle. A 1.5x factor applied once per iteration
        can traverse the entire allowed range in ~10 iterations, so a single noisy
        KL reading strands the learning rate at a bound for the rest of the run.
        """
        low, high = self.config.lr_bounds
        step = self.config.lr_adapt_factor
        if approx_kl > self.config.target_kl * 2.0:
            self.learning_rate = max(low, self.learning_rate / step)
        elif 0.0 < approx_kl < self.config.target_kl / 2.0:
            self.learning_rate = min(high, self.learning_rate * step)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate
