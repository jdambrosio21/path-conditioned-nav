"""Path-conditioned RL local planning for long-range navigation (wheeled robot).

Reimplementation of Haro et al., "Path-conditioned Reinforcement Learning-based
Local Planning for Long-Range Navigation" (arXiv:2603.13888), retargeted from an
Isaac Sim / B2W quadruped setup to a wheeled platform that trains on Apple
Silicon. See README.md for the full list of deviations from the paper.
"""

from .config import EnvConfig, ExperimentConfig, PolicyConfig, PPOConfig, TrainConfig
from .planning import PathQuality

__all__ = [
    "EnvConfig",
    "ExperimentConfig",
    "PolicyConfig",
    "PPOConfig",
    "TrainConfig",
    "PathQuality",
]
