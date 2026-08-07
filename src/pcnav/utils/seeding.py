"""Deterministic seeding across numpy and torch."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed every RNG the project touches.

    Note that full bitwise determinism is not achievable on the MPS backend; this
    makes runs comparable, not byte-identical.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
