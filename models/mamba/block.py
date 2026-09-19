import torch.nn as nn


class EMambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        expand,
        d_state,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

    def forward(self, x):
        raise NotImplementedError
