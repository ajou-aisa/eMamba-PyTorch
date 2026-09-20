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

    def __init__(self, d_inner: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_conv = 4

        self.conv1d = nn.Conv1d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            kernel_size = self.d_conv,
            groups = self.d_inner,
            padding = self.d_conv - 1,
            bias = True,
        )

    def forward(self, tokens: Tensor) -> Tensor:
        # TODO:
        # Apply the 1D convolution shown in Figure 3 over the token sequence.
        #
        # The paper specifies this stage but does not provide enough detail
        # to fix kernel size, padding, grouping, or boundary handling.
        #
        # Interface:
        #   [B, L, ED] -> [B, L, ED]

        seq_len = tokens.shape[1]

        hidden = tokens.transpose(1, 2)     #[B, ED, L]
        hidden = self.conv1d(hidden)        #[B, ED, L + 3]
        hidden = hidden[..., :seq_len]      #[B, ED, L]
        hidden = hidden.transpose(1, 2)     #[B, L, ED]
        
        return hidden

        raise NotImplementedError(
            "MambaConv1D configuration is not specified yet."
        )