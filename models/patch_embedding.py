"""Input stage of Figures 3 and 5."""

from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """Interface: [B, H, W, C] -> [B, L, D]."""

    def __init__(self, in_channels: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        if in_channels <= 0 or patch_size <= 0 or d_model <= 0:
            raise ValueError("in_channels, patch_size, and d_model must be positive")
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model

        patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(patch_dim, d_model, bias=True)

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4:
            raise ValueError(
                f"frames.ndim {frames.ndim} != 4"
            )

        B, H, W, C = frames.shape
        P = self.patch_size

        if C != self.in_channels:
            raise ValueError(
                f"C {C} != in_channels {self.in_channels}"
            )

        if H % P != 0 or W % P != 0:
            raise ValueError(
                f"H % P {H % P} != 0 or W % P {W % P} != 0"
            )

        # [B, H, W, C]
        # -> [B, H/P, P, W/P, P, C]
        patches = frames.reshape(
            B,
            H // P,
            P,
            W // P,
            P,
            C,
        )

        # [B, H/P, P, W/P, P, C]
        # ->         # [B, H/P, P, W/P, P, C]
        # -> [B, H/P, W/P, P, P, C]
        patches = patches.permute(
            0, 1, 3, 2, 4, 5
        )

        # Number of patches
        L = (H // P) * (W // P)

        # [B, H/P, W/P, P, P, C] -> [B, L, P*P*C]
        tokens = patches.reshape(
            B,
            L,
            -1,
        )

        # [B, L, P*P*C] -> [B, L, D]
        return self.proj(tokens)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, patch_size={self.patch_size}, "
            f"d_model={self.d_model}"
        )