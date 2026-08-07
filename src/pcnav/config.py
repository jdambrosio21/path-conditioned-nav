"""Central configuration.

Every tunable in the project is declared here as a frozen-by-convention dataclass
so that a run is fully described by one serializable object. `ExperimentConfig`
is what gets written next to a checkpoint, and what `scripts/evaluate.py` reloads
to guarantee the evaluation env matches the training env.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Physical constants of the wheeled platform (B2W-scale)
# --------------------------------------------------------------------------
ARENA_SIZE_M = 30.0          # paper's training arena edge length
GRID_RESOLUTION_M = 0.25     # occupancy grid / geodesic field cell size
ROBOT_RADIUS_M = 0.35        # footprint half-width, used to inflate obstacles

MAX_FORWARD_SPEED = 2.0      # m/s
MAX_REVERSE_SPEED = 0.5      # m/s
MAX_YAW_RATE = 1.5           # rad/s
MAX_LINEAR_ACCEL = 3.0       # m/s^2
MAX_ANGULAR_ACCEL = 6.0      # rad/s^2

# --------------------------------------------------------------------------
# Control timing -- mirrors the paper's two-rate architecture
# --------------------------------------------------------------------------
NAV_POLICY_HZ = 5.0          # high-level navigation policy
LOW_LEVEL_HZ = 50.0          # frozen velocity-tracking controller
SUBSTEPS_PER_ACTION = int(LOW_LEVEL_HZ / NAV_POLICY_HZ)
SIM_TIMESTEP = 1.0 / LOW_LEVEL_HZ

# --------------------------------------------------------------------------
# Perception -- 2-D analogue of the paper's 40x64 ZED X depth image
# --------------------------------------------------------------------------
NUM_RAYS = 64
SENSOR_FOV_RAD = float(np.deg2rad(105.0))   # paper's horizontal FOV
SENSOR_MAX_RANGE_M = 10.0                   # paper's max depth range

# --------------------------------------------------------------------------
# Reference path discretization
# --------------------------------------------------------------------------
PATH_VERTEX_SPACING_M = 0.25   # polyline resample spacing
MAX_PATH_VERTICES = 256        # padded length (= 64 m of path)
NUM_WAYPOINTS = 15             # waypoints exposed to the policy (paper)
WAYPOINT_SPACING_M = 1.0       # arclength between consecutive waypoints

# --------------------------------------------------------------------------
# Episode termination
# --------------------------------------------------------------------------
GOAL_RADIUS_M = 1.0
EPISODE_DURATION_S = 60.0      # paper's training episode length
MAX_EPISODE_STEPS = int(EPISODE_DURATION_S * NAV_POLICY_HZ)

# Derived tensor dimensions.
OBS_DIM = NUM_RAYS + 5 + 4 + 1   # scan + goal encoding + proprioception + has_path
PRIV_DIM = 4                     # critic-only scalars
ACTION_DIM = 2                   # (forward velocity, yaw rate)
WAYPOINT_FEATURES = 3            # (x_body, y_body, valid)


@dataclass
class RewardConfig:
    """Reward weights.

    Deliberately contains **no path-following term**. The paper's central claim is
    that conditioning on the path as an *observation*, while rewarding only
    goal-reaching, yields a policy that exploits good paths and ignores bad ones.
    Rewarding path adherence directly would destroy exactly that property.
    """

    progress: float = 1.0        # per-metre of true geodesic progress toward goal
    goal_bonus: float = 50.0     # terminal success
    shortcut: float = 0.6        # the paper's novel opportunistic-deviation term
    collision: float = -25.0     # terminal failure
    clearance: float = -0.4      # soft penalty for hugging obstacles
    action_rate: float = -0.02   # motion smoothness regularization
    time: float = -0.02          # per-step cost, encourages efficiency
    clearance_threshold_m: float = 0.8   # penalty engages below this clearance


@dataclass
class MapConfig:
    """Procedural arena generation."""

    num_maps: int = 180                       # paper trains across 180 arenas
    arena_size_m: float = ARENA_SIZE_M
    num_obstacles: tuple[int, int] = (18, 45)
    obstacle_radius_m: tuple[float, float] = (0.30, 1.60)
    goals_per_map: int = 8
    starts_per_goal: int = 12
    min_start_goal_geodesic_m: float = 12.0   # keeps every episode long-range
    roadmap_nodes: int = 400
    roadmap_neighbors: int = 10


@dataclass
class EnvConfig:
    """Environment instantiation."""

    num_envs: int = 4096
    device: str = "mps"
    seed: int = 0
    maps: MapConfig = field(default_factory=MapConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    # Training mixture over reference-path conditions. Enough corrupted and absent
    # mass that the policy cannot learn to trust the path unconditionally.
    path_quality_mix: dict[str, float] = field(
        default_factory=lambda: {
            "OPTIMAL": 0.30,
            "NOISY": 0.25,
            "SUBOPTIMAL": 0.20,
            "DETOURED": 0.10,
            "WRONG_GOAL": 0.05,
            "NONE": 0.10,
        }
    )
    # Pin a single condition instead of sampling the mixture (ablation mode).
    fixed_path_quality: str | None = None


@dataclass
class PolicyConfig:
    """Actor-critic architecture.

    The path encoder follows the paper: self-attention across the waypoint
    sequence, then cross-attention from the robot-state embedding into it.
    """

    # Sized against the paper, which reports a 1.76M-parameter actor of which only
    # 12,960 parameters belong to the path-encoding module. The encoder is meant to
    # be small: it summarizes 15 waypoints, not an image. Our observation is 74-D
    # rather than their 2636-D flattened depth map, so the trunk shrinks to match.
    embed_dim: int = 64
    num_heads: int = 4
    encoder_layers: int = 1
    trunk_hidden: tuple[int, ...] = (256, 128)
    init_log_std: float = -0.5
    log_std_bounds: tuple[float, float] = (-3.0, 0.0)
    # Temporally consistent dropout: resample the mask once per episode rather
    # than per step, so the policy sees a coherent perturbation over time.
    dropout: float = 0.1
    temporally_consistent_dropout: bool = True


@dataclass
class PPOConfig:
    """Asymmetric-critic PPO, following the rsl-rl defaults the paper uses."""

    rollout_steps: int = 24
    num_epochs: int = 5
    num_minibatches: int = 4
    clip_ratio: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 5e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
    learning_rate: float = 1e-3
    # rsl-rl style adaptive schedule: nudge the LR to hold KL near target.
    adaptive_lr: bool = True
    target_kl: float = 0.01
    lr_adapt_factor: float = 1.1   # gentle; 1.5 traverses the whole LR range too fast
    lr_bounds: tuple[float, float] = (1e-5, 1e-2)


@dataclass
class TrainConfig:
    total_iterations: int = 3000
    log_interval: int = 10
    checkpoint_interval: int = 200
    run_dir: str = "runs"
    run_name: str = "pcnav"
    # Path to a checkpoint to warm-start from. The paper builds on a pretrained
    # navigation base rather than starting from random weights.
    init_from: str | None = None


@dataclass
class ExperimentConfig:
    """Full description of one run. Serialized alongside every checkpoint."""

    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Rebuild from a serialized dict, restoring nested dataclasses."""
        return cls(
            env=EnvConfig(
                **{**data["env"],
                   "maps": MapConfig(**data["env"]["maps"]),
                   "reward": RewardConfig(**data["env"]["reward"])}
            ),
            policy=PolicyConfig(**data["policy"]),
            ppo=PPOConfig(**data["ppo"]),
            train=TrainConfig(**data["train"]),
        )
