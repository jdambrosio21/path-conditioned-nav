"""Environment backends: fast batched PyTorch, and MuJoCo for validation/viewing."""

from .torch_env import PathConditionedNavEnv

__all__ = ["PathConditionedNavEnv"]

# MuJoCo backend is imported lazily by name to keep training free of the dependency.
def make_mujoco_env(*args, **kwargs):
    """Construct the MuJoCo-backed environment (evaluation / visualization only)."""
    from .mujoco_env import MuJoCoNavEnv

    return MuJoCoNavEnv(*args, **kwargs)
