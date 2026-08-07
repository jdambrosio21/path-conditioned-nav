"""Training loop: rollout collection, PPO updates, logging, checkpointing."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ..config import ACTION_DIM, OBS_DIM, ExperimentConfig
from ..envs.torch_env import PathConditionedNavEnv
from ..models.actor_critic import PathConditionedActorCritic
from ..utils.logging import EpisodeTracker, RunLogger
from .ppo import PPO, RolloutBuffer


class Runner:
    """Owns the env, policy, optimizer and rollout buffer for one run."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.env.device)

        self.env = PathConditionedNavEnv(config.env)
        self.policy = PathConditionedActorCritic(config.policy, config.env.num_envs).to(self.device)
        self.algorithm = PPO(self.policy, config.ppo, self.device)

        observation = self.env.observe()
        # The dropout mask is stored alongside the observation so that the update
        # can replay the exact network the rollout action was drawn from.
        obs_shapes = {k: tuple(v.shape[1:]) for k, v in observation.items()}
        obs_shapes["dropout_mask"] = (OBS_DIM,)
        self.buffer = RolloutBuffer(
            num_steps=config.ppo.rollout_steps,
            num_envs=config.env.num_envs,
            obs_shapes=obs_shapes,
            action_dim=ACTION_DIM,
            device=self.device,
        )

        self.tracker = EpisodeTracker()
        self.run_dir = Path(config.train.run_dir) / config.train.run_name
        self.logger = RunLogger(self.run_dir)
        (self.run_dir / "config.json").write_text(
            json.dumps(config.to_dict(), indent=2, default=str)
        )

        if config.train.init_from:
            self.load_policy_weights(Path(config.train.init_from))

        self.total_env_steps = 0
        self._episode_return = torch.zeros(config.env.num_envs, device=self.device)
        self._episode_length = torch.zeros(config.env.num_envs, device=self.device)

    # --------------------------------------------------------------- rollout

    def _attach_dropout_mask(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Tag an observation with the dropout mask currently in effect."""
        return {**observation, "dropout_mask": self.policy.obs_dropout.current_mask.clone()}

    def collect_rollout(self, observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Run `rollout_steps` environment steps, filling the buffer."""
        self.buffer.clear()

        for _ in range(self.config.ppo.rollout_steps):
            observation = self._attach_dropout_mask(observation)
            step_output = self.policy.act(observation)
            next_observation, reward, done, info = self.env.step(step_output["action"])

            self.buffer.add(observation, step_output, reward, done)

            self._episode_return += reward
            self._episode_length += 1
            self.tracker.record(done, info, self._episode_return, self._episode_length)

            # Reset per-episode accumulators for environments that just finished,
            # and give them fresh temporally-consistent dropout masks.
            if bool(done.any()):
                finished = done.nonzero(as_tuple=True)[0]
                self._episode_return[finished] = 0.0
                self._episode_length[finished] = 0.0
                self.policy.obs_dropout.resample(finished)

            observation = next_observation
            self.total_env_steps += self.config.env.num_envs

        return self._attach_dropout_mask(observation)

    # ---------------------------------------------------------------- training

    def train(self) -> None:
        config = self.config
        observation = self.env.observe()
        self.policy.obs_dropout.resample()

        for iteration in range(1, config.train.total_iterations + 1):
            observation = self.collect_rollout(observation)

            with torch.no_grad():
                last_value = self.policy.value(
                    observation["obs"],
                    observation["priv"],
                    observation["opt_path"],
                    observation["path"],
                )
            self.buffer.compute_returns(last_value, config.ppo.gamma, config.ppo.gae_lambda)

            stats = self.algorithm.update(self.buffer)

            if iteration % config.train.log_interval == 0:
                self.logger.log(
                    iteration,
                    self.total_env_steps,
                    {
                        **stats,
                        "mean_return": self.tracker.mean_return,
                        "mean_length": self.tracker.mean_length,
                        "success_by_quality": self.tracker.success_by_quality(),
                        "length_by_quality": self.tracker.length_by_quality(),
                        "termination_by_quality": self.tracker.termination_breakdown(),
                    },
                )

            if iteration % config.train.checkpoint_interval == 0:
                self.save_checkpoint(iteration)

        self.save_checkpoint(config.train.total_iterations, final=True)

    # ------------------------------------------------------------ checkpoints

    def load_policy_weights(self, checkpoint_path: Path) -> None:
        """Warm-start from a previously trained policy.

        The paper does not train from scratch: it extends a pre-trained navigation
        base (Yang et al. 2025, including a pretrained depth encoder) and learns
        the path-encoding module on top. Learning obstacle avoidance, goal seeking
        and path-quality discrimination simultaneously from random weights is a
        materially harder problem than the one the paper solves.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        missing, unexpected = self.policy.load_state_dict(checkpoint["policy"], strict=False)
        print(
            f"warm-started from {checkpoint_path} "
            f"({checkpoint.get('env_steps', 0):,} env steps; "
            f"{len(missing)} missing, {len(unexpected)} unexpected keys)",
            flush=True,
        )

    def save_checkpoint(self, iteration: int, final: bool = False) -> None:
        name = "policy_final.pt" if final else f"policy_{iteration:06d}.pt"
        torch.save(
            {
                "iteration": iteration,
                "env_steps": self.total_env_steps,
                "policy": self.policy.state_dict(),
                "config": self.config.to_dict(),
            },
            self.run_dir / name,
        )
