"""Range normalization boundary from Section 4.1."""

from torch import Tensor, nn


class RangeNorm(nn.Module):
    """Interface: [B, L, D] -> [B, L, D], normalized over D."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, tokens: Tensor) -> Tensor:
        # TODO:
        # 1. Compute the mean over the feature dimension.
        # 2. Center the input with x - mean(x).
        # 3. Compute range = max(x) - min(x).
        # 4. Normalize by the range and apply learnable gamma and beta.
        #
        # y = gamma * (x - mean) / range(x - mean) + beta
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}"