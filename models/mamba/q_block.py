import torch
from torch import Tensor, nn

from ptq.quant import QuantRuntime
from ptq.piecewise import piecewise_silu


class QBlock(nn.Module):
    def __init__(
        self, norm: nn.Module, gate_proj: nn.Module, input_proj: nn.Module,
        conv: nn.Module, ssm: nn.Module, output_proj: nn.Module,
        d_model: int, prefix: str, runtime: QuantRuntime,
    ) -> None:
        super().__init__()
        self.norm = norm
        self.gate_proj = gate_proj
        self.input_proj = input_proj
        self.conv = conv
        self.ssm = ssm
        self.output_proj = output_proj
        self.d_model = d_model
        self.prefix = prefix
        self.runtime = runtime

    def forward(self, tokens: Tensor) -> Tensor:

        # fp32 -> int8 -> fp32
        residual = self.runtime.boundary(f"{self.prefix}.residual", tokens)

        # QRangeNorm
        normalized = self.norm(tokens)

        # Upper path: projection(QLinear) -> SiLU.
        gate = self.gate_proj(normalized)
        gate = piecewise_silu(gate) if self.runtime.use_pwl else torch.nn.functional.silu(gate)
        gate = self.runtime.boundary(f"{self.prefix}.gate", gate)

        # Lower path: projection(QLinear) -> convolution(QConv1d) -> SSM(QSelectiveSSM).
        hidden = self.input_proj(normalized)
        hidden = self.runtime.boundary(f"{self.prefix}.conv", self.conv(hidden))
        hidden = self.ssm(hidden)


        hidden = self.runtime.boundary(f"{self.prefix}.gated", hidden * gate)

        # QLinear -> int8 -> fp32
        hidden = self.output_proj(hidden)

        # add residual -> quantize -> fp32
        return self.runtime.boundary(f"{self.prefix}.output", residual + hidden)
