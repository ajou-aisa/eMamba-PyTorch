from fractions import Fraction

import torch
from torch import Tensor, nn

from ptq.ops import INT64_MAX, align_codes, checked_add, checked_mul, dequantize_codes, saturate
from ptq.quant import QuantRuntime

from .q_range_norm_math import NORM_FRACTION_BITS, left_shift_checked, round_ratio


# QRangeNorm: nn.RangeNorm -> QRangeNorm
class QRangeNorm(nn.Module):
    def __init__(
        self, gamma: Tensor, beta: Tensor, eps: float, prefix: str,
        runtime: QuantRuntime,
    ) -> None:
        super().__init__()
        self.gamma = nn.Parameter(gamma.detach().clone(), requires_grad=False)
        self.beta = nn.Parameter(beta.detach().clone(), requires_grad=False)
        self.eps, self.prefix, self.runtime = eps, prefix, runtime
        self.epsilon_code = 1
        if runtime.is_frozen:
            effective = float(torch.tensor(eps, dtype=torch.float32).item())
            entry = runtime.entry(f"{prefix}.input")
            epsilon_units = Fraction.from_float(effective) / Fraction(2) ** entry.exponent
            self.epsilon_code = min(
                INT64_MAX, max(1, round(epsilon_units * (1 << NORM_FRACTION_BITS))),
            )

    # for calibration
    def _float_forward(self, tokens: Tensor) -> Tensor:
        gamma = self.runtime.boundary(f"{self.prefix}.gamma", self.gamma)
        beta = self.runtime.boundary(f"{self.prefix}.beta", self.beta)
        mean = tokens.mean(dim=-1, keepdim=True)
        centered = tokens - mean
        value_range = (
            centered.amax(dim=-1, keepdim=True)
            - centered.amin(dim=-1, keepdim=True)
        ).clamp_min(self.eps)
        normalized = centered / value_range
        return gamma * normalized + beta

    # for ptq frozen model
    def _integer_forward(self, tokens: Tensor) -> Tensor:
        output_entry = self.runtime.entry(f"{self.prefix}.output")
        gamma_entry, beta_entry = (self.runtime.entry(f"{self.prefix}.{name}")
                                   for name in ("gamma", "beta"))
        x = self.runtime.codes(f"{self.prefix}.input", tokens)
        gamma_name, beta_name = f"{self.prefix}.gamma", f"{self.prefix}.beta"
        gamma = self.runtime.codes(gamma_name, self.runtime.boundary(gamma_name, self.gamma))
        beta = self.runtime.codes(beta_name, self.runtime.boundary(beta_name, self.beta))

        total = left_shift_checked(x.sum(dim=-1, keepdim=True), NORM_FRACTION_BITS)
        mean = round_ratio(total, torch.full_like(total, x.shape[-1]))
        centered = left_shift_checked(x, NORM_FRACTION_BITS) - mean
        value_range = (
            centered.amax(dim=-1, keepdim=True)
            - centered.amin(dim=-1, keepdim=True)
        ).clamp_min(self.epsilon_code)
        normalized = round_ratio(left_shift_checked(centered, NORM_FRACTION_BITS), value_range)
        scaled = checked_mul(gamma, normalized)
        scaled_exponent = gamma_entry.exponent - NORM_FRACTION_BITS
        common = min(scaled_exponent, beta_entry.exponent)
        output = checked_add(
            align_codes(scaled, scaled_exponent, common),
            align_codes(beta, beta_entry.exponent, common),
        )
        result = align_codes(output, common, output_entry.exponent)
        return dequantize_codes(saturate(result, 8), output_entry.exponent, tokens.dtype)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = self.runtime.boundary(f"{self.prefix}.input", tokens)
        result = self._integer_forward(tokens) if self.runtime.is_frozen else self._float_forward(tokens)
        return self.runtime.boundary(f"{self.prefix}.output", result)
