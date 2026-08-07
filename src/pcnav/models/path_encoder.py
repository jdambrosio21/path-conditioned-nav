"""Reference-path encoder.

Follows the paper's two-stage attention design: self-attention across the
waypoint sequence to build path-internal context, then cross-attention from the
robot-state embedding into that sequence to extract the part of the path that is
relevant right now.

The critical detail is masking. A waypoint is invalid when it lies past the end
of the path, and *every* waypoint is invalid when the policy was handed no path
at all -- which happens on 10% of training episodes by design. Attention over a
fully-masked sequence produces NaNs, so invalid rows are given a dummy attendable
slot and their output is explicitly zeroed. That zero is what teaches the policy
"no path information available", and it must be a clean zero rather than garbage.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import WAYPOINT_FEATURES


class TemporallyConsistentDropout(nn.Module):
    """Dropout whose mask is resampled per episode rather than per forward pass.

    Standard dropout injects fresh noise every timestep, which a recurrent-ish
    control policy can average away across an episode. Holding the mask fixed for
    the whole episode instead forces the policy to stay robust to a *persistent*
    missing subset of features -- the regularizer the paper reports using.
    """

    def __init__(self, num_envs: int, num_features: int, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob
        self.register_buffer(
            "mask", torch.ones(num_envs, num_features), persistent=False
        )

    @torch.no_grad()
    def resample(self, env_ids: torch.Tensor | None = None) -> None:
        """Draw new masks, for the given environments or all of them."""
        if self.drop_prob <= 0.0:
            return
        if env_ids is None:
            env_ids = torch.arange(self.mask.shape[0], device=self.mask.device)
        if env_ids.numel() == 0:
            return
        keep_prob = 1.0 - self.drop_prob
        drawn = (
            torch.rand(env_ids.numel(), self.mask.shape[1], device=self.mask.device) < keep_prob
        )
        self.mask[env_ids] = drawn.float() / keep_prob   # inverted dropout scaling

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Apply a dropout mask.

        `mask` must be supplied when re-evaluating stored transitions: the mask
        used at collection time is recorded in the rollout buffer and replayed
        here. Reusing the *current* mask instead would mean the update scores the
        actions under a different network than the one that produced them, which
        silently corrupts the PPO importance ratio.
        """
        if self.drop_prob <= 0.0:
            return x
        return x * (self.mask if mask is None else mask)

    @property
    def current_mask(self) -> torch.Tensor:
        return self.mask


class WaypointEncoder(nn.Module):
    """Self-attention over waypoints, then cross-attention from the robot state."""

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, num_layers: int = 2):
        """Note the deliberate absence of dropout inside the attention stack.

        Standard ``nn.Dropout`` redraws its mask on every forward pass, so a
        transition re-scored during a PPO update would run through a *different*
        network than the one that produced the action, silently corrupting the
        importance ratio. All stochastic regularization in this project therefore
        lives in :class:`TemporallyConsistentDropout`, whose mask is recorded in
        the rollout buffer and replayed at update time.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.waypoint_proj = nn.Linear(WAYPOINT_FEATURES, embed_dim)

        self.self_attn_layers = nn.ModuleList(
            nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        )
        self.self_attn_norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(num_layers))
        self.feedforwards = nn.ModuleList(
            nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim), nn.GELU(), nn.Linear(2 * embed_dim, embed_dim)
            )
            for _ in range(num_layers)
        )
        self.feedforward_norms = nn.ModuleList(nn.LayerNorm(embed_dim) for _ in range(num_layers))

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, waypoints: torch.Tensor, state_embedding: torch.Tensor) -> torch.Tensor:
        """Encode a waypoint sequence into a single state-conditioned context vector.

        Args:
            waypoints:       (B, N, 3) body-frame (x, y, valid) waypoint tokens.
            state_embedding: (B, D) embedding of the robot's own observation.

        Returns:
            (B, D) path context. Exactly zero for environments with no valid path.
        """
        valid = waypoints[..., 2] > 0.5                     # (B, N)
        has_any_path = valid.any(dim=1, keepdim=True)       # (B, 1)

        # Give fully-masked rows one attendable slot so attention stays finite;
        # their output is zeroed at the end anyway.
        attendable = valid.clone()
        attendable[:, 0] |= ~valid.any(dim=1)
        key_padding_mask = ~attendable                      # True == ignore

        tokens = self.waypoint_proj(waypoints)
        for attn, attn_norm, ff, ff_norm in zip(
            self.self_attn_layers,
            self.self_attn_norms,
            self.feedforwards,
            self.feedforward_norms,
            strict=True,
        ):
            attended, _ = attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
            tokens = attn_norm(tokens + attended)
            tokens = ff_norm(tokens + ff(tokens))

        query = state_embedding.unsqueeze(1)                # (B, 1, D)
        context, _ = self.cross_attn(query, tokens, tokens, key_padding_mask=key_padding_mask)
        context = self.output_norm(context.squeeze(1))

        # Hard zero for "no path", so the absence of guidance is unambiguous.
        return context * has_any_path.float()
