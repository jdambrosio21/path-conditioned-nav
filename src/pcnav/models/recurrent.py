"""Spatially-Enhanced Recurrent Unit (SRU).

From Yang et al., *Spatially-Enhanced Recurrent Memory for Long-Range Mapless
Navigation via End-to-End Reinforcement Learning* (arXiv:2506.05997) — the base
navigation policy that the path-conditioning paper builds on.

The idea, and why it is more than a tweak
-----------------------------------------
A standard GRU combines the current observation with memory **additively**::

    h̃_t = tanh(W_xh·x_t + W_hh·(r_t ⊙ h_{t-1}) + b_h)

Additive mixing cannot cheaply express *"re-align my memory with what I am
looking at now."* Spatial alignment between an egocentric observation and a
stored map is fundamentally a transformation — bilinear in (observation, memory)
— and an additive layer has to approximate it with a great deal of capacity.

The SRU adds one multiplicative term::

    s_t = W_xs·x_t + b_s
    h̃_t = tanh( s_t ⊙ (W_xh·x_t + W_hh·(r_t ⊙ h_{t-1}) + b_h) )

`s_t` is derived from the current observation and gates the candidate state
element-wise, so the observation can *scale* the contribution of each memory
channel rather than merely add to it. That single bilinear interaction is what
lets the unit learn implicit spatial transformations. The authors report a 23.5%
improvement over LSTM/GRU on long-range navigation.

Note there is no explicit pose tracking and no geometric grid. Odometry enters
only as ordinary proprioceptive input; the alignment is learned.

Sequence handling
-----------------
Reinforcement learning makes recurrence delicate. The hidden state must be reset
at episode boundaries, and — critically — the update must replay each rollout
through the *same* hidden states that produced the actions. Reusing whatever
hidden state happens to be current would score the stored actions under a
different network than generated them, silently corrupting the PPO importance
ratio. This is the same failure mode that the dropout mask already caused in this
codebase, one level harder, so `forward_sequence` takes the initial hidden state
and the done flags explicitly rather than holding state internally.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpatiallyEnhancedGRUCell(nn.Module):
    """One SRU-GRU step: a GRU cell with a multiplicative spatial term."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Update (z) and reset (r) gates -- identical to a standard GRU.
        self.input_to_gates = nn.Linear(input_size, 2 * hidden_size)
        self.hidden_to_gates = nn.Linear(hidden_size, 2 * hidden_size, bias=False)

        # Candidate state.
        self.input_to_candidate = nn.Linear(input_size, hidden_size)
        self.hidden_to_candidate = nn.Linear(hidden_size, hidden_size, bias=False)

        # The spatial transformation term: the whole difference from a GRU.
        self.input_to_spatial = nn.Linear(input_size, hidden_size)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.input_to_gates,
            self.hidden_to_gates,
            self.input_to_candidate,
            self.hidden_to_candidate,
        ):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        # Initialize the spatial term near unity so the cell starts out behaving
        # like a plain GRU and learns to depart from it, rather than starting from
        # a randomly scaled candidate state that would stall early learning.
        nn.init.zeros_(self.input_to_spatial.weight)
        nn.init.ones_(self.input_to_spatial.bias)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Advance one timestep.

        Args:
            x:      (B, input_size) current observation features.
            hidden: (B, hidden_size) previous hidden state.
        Returns:
            (B, hidden_size) new hidden state.
        """
        gates = self.input_to_gates(x) + self.hidden_to_gates(hidden)
        update_gate, reset_gate = gates.chunk(2, dim=-1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)

        spatial = self.input_to_spatial(x)
        candidate_pre = self.input_to_candidate(x) + self.hidden_to_candidate(
            reset_gate * hidden
        )
        candidate = torch.tanh(spatial * candidate_pre)

        return (1.0 - update_gate) * candidate + update_gate * hidden


class RecurrentMemory(nn.Module):
    """Stacked SRU cells with explicit, externally-owned hidden state."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            SpatiallyEnhancedGRUCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single timestep across all layers.

        Args:
            x:      (B, input_size)
            hidden: (num_layers, B, hidden_size)
        Returns:
            (output, new_hidden) where output is the top layer's (B, hidden_size).
        """
        new_hidden = []
        layer_input = x
        for layer, cell in enumerate(self.cells):
            layer_input = cell(layer_input, hidden[layer])
            new_hidden.append(layer_input)
        return layer_input, torch.stack(new_hidden, dim=0)

    def forward_sequence(
        self,
        x: torch.Tensor,
        initial_hidden: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Replay a whole rollout, resetting the hidden state at episode ends.

        Args:
            x:              (T, B, input_size) features in timestep order.
            initial_hidden: (num_layers, B, hidden_size) state entering step 0.
            dones:          (T, B) 1.0 where the episode ended *at* that step.
        Returns:
            (T, B, hidden_size) top-layer outputs.

        The reset is applied **after** producing the output for step t, because the
        action at step t was taken before the environment terminated. Zeroing
        beforehand would attribute a fresh episode's memory to the final action of
        the previous one.
        """
        timesteps = x.shape[0]
        hidden = initial_hidden
        outputs = []
        for t in range(timesteps):
            output, hidden = self.forward(x[t], hidden)
            outputs.append(output)
            hidden = hidden * (1.0 - dones[t]).view(1, -1, 1)
        return torch.stack(outputs, dim=0)
