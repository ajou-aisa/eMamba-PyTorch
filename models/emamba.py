"""eMamba composition: patch embedding -> Mamba blocks -> output head."""

from torch import Tensor, nn

from .mamba import EMambaBlock
from .output_head import OutputHead
from .patch_embedding import PatchEmbedding


class EMamba(nn.Module):
    """MARS interface: [B, 8, 8, 5] -> [B, 57]."""

    def __init__(
        self,
        d_model: int = 20,
        expand: int = 2,
        patch_size: int = 2,
        num_blocks: int = 2,
        d_state: int = 8,
        out_dim: int = 57,
        in_channels: int = 5,
        readout: str = "flatten",
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        self.d_model = d_model
        self.expand = expand
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.d_state = d_state

        self.patch_embedding = PatchEmbedding(
            in_channels = in_channels,
            patch_size = patch_size,
            d_model = d_model,
        )
        self.blocks = nn.ModuleList([
            EMambaBlock(d_model = d_model, expand = expand, d_state = d_state)
            for _ in range(num_blocks)
        ])
        self.head = OutputHead(
            d_model=d_model, out_dim=out_dim, readout=readout,
            num_tokens=(8 // patch_size) ** 2,
        )

    def forward(self, frames: Tensor) -> Tensor:
        tokens = self.patch_embedding(frames)

        for block in self.blocks:
            tokens = block(tokens)

        return self.head(tokens)
