"""Mamba block connectivity from Figure 3."""

from torch import Tensor, nn

from .range_norm import RangeNorm
from .selective_ssm import SelectiveSSM


class EMambaBlock(nn.Module):
    """Interface: [B, L, D] -> [B, L, D]; internal width is ED."""

    def __init__(self, d_model: int, expand: int, d_state: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.norm = RangeNorm(d_model)

        # Upper path: projection -> SiLU.
        self.gate_proj = nn.Linear(d_model, self.d_inner)
        self.gate_act = nn.SiLU()

        # Lower path: projection -> convolution -> SSM.
        self.input_proj = nn.Linear(d_model, self.d_inner)
        self.conv = MambaConv1D(self.d_inner)
        self.ssm = SelectiveSSM(self.d_inner, d_state)

        self.output_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        residual = tokens
        normalized = self.norm(tokens)

        # Upper path
        gate = self.gate_proj(normalized)
        gate = self.gate_act(gate)

        # Lower path
        hidden = self.input_proj(normalized)
        hidden = self.conv(hidden)
        hidden = self.ssm(hidden)

        hidden = hidden * gate
        hidden = self.output_proj(hidden)

        return residual + hidden


class MambaConv1D(nn.Module):
    """1D convolution stage in the lower Mamba path."""

    def __init__(self, d_inner: int, d_conv: int = 4) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_conv = d_conv

        self.conv1d = nn.Conv1d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            kernel_size = self.d_conv,
            groups = self.d_inner,
            padding = self.d_conv - 1,
            bias = True,
        )

    def forward(self, tokens: Tensor) -> Tensor:

        # Interface: [B, L, ED] -> [B, L, ED]
        seq_len = tokens.size(1)

        #[B, L, ED] -> [B, ED, L]
        hidden = tokens.transpose(1, 2).contiguous()

        #[B, ED, L] -> [B, ED, L + d_conv - 1]
        hidden = self.conv1d(hidden)

        hidden = hidden[..., :seq_len].contiguous()

        #[B, ED, L] -> [B, L, ED]
        hidden = hidden.transpose(1, 2).contiguous()
        
        return hidden