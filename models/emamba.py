import torch.nn as nn

from .emamba_block import EMambaBlock


class EMamba(nn.Module):
    def __init__(
        self,
        d_model=20,
        expand=2,
        patch_size=2,
        num_blocks=2,
        d_state=8,
        out_dim=57,
    ):
        super().__init__()

        self.d_model = d_model
        self.expand = expand
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.d_state = d_state

        # Mamba blocks
        self.blocks = nn.ModuleList([
            EMambaBlock(
                d_model=d_model,
                expand=expand,
                d_state=d_state,
            )
            for _ in range(num_blocks)
        ])

        # 19 joints × (x, y, z)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        raise NotImplementedError
