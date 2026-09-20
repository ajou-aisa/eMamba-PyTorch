"""Range normalization boundary from Section 4.1."""

import torch
from torch import Tensor, nn


class RangeNorm(nn.Module):
    """Interface: [B, L, D] -> [B, L, D], normalized over D."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # Learnable scale and shift.
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, tokens: Tensor) -> Tensor:
            # Mean of each token over the feature dimension D.
            mean = tokens.mean(dim=-1, keepdim=True)

            # Center the input.
            centered = tokens - mean

            # Range of each token.
            x_max = centered.amax(dim=-1, keepdim=True)
            x_min = centered.amin(dim=-1, keepdim=True)
            value_range = (x_max - x_min).clamp_min(self.eps)

            # Range normalization.
            normalized = centered / value_range

            # Learnable scale and shift.
            return self.gamma * normalized + self.beta

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"