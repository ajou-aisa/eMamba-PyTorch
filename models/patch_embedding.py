"""Input stage of Figures 3 and 5."""

from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Interface: [B, H, W, C] -> [B, L, D]."""

    def __init__(self, in_channels: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model

    def forward(self, frames: Tensor) -> Tensor:
        # TODO:
        # 1. Split the H x W input into non-overlapping P x P patches.
        # 2. Flatten each patch into one token.
        # 3. Return a token sequence with shape [B, L, D].
        #
        # For MARS: [B, 8, 8, 5] -> [B, 16, 20].
        # The paper does not specify whether an additional learnable
        # projection is used after patch extraction.
        raise NotImplementedError

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, patch_size={self.patch_size}, "
            f"d_model={self.d_model}"
        )