import math
from typing import Final

import torch
from torch import Tensor
from torch.nn import functional as F

SILU_KNOTS: Final = (-7.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
EXP_KNOTS: Final = (-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                   0.0, 0.25, 0.5, 1.0)
PIECEWISE_SPEC: Final = {
    "method": "secant_fp32_v1", "silu_knots": list(SILU_KNOTS), "exp_knots": list(EXP_KNOTS),
    "silu_tails": "zero_identity", "exp_tails": "zero_e",
}


def _interpolate(value: Tensor, boundaries: Tensor, ordinates: Tensor) -> Tensor:
    indices = (torch.bucketize(value.detach().contiguous(), boundaries) - 1).clamp(
        0, boundaries.numel() - 2,
    )
    x0, x1 = boundaries[indices], boundaries[indices + 1]
    y0, y1 = ordinates[indices], ordinates[indices + 1]
    return y0 + (value - x0) * (y1 - y0) / (x1 - x0)


def piecewise_silu(value: Tensor) -> Tensor:
    boundaries = value.new_tensor(SILU_KNOTS)
    result = _interpolate(value, boundaries, F.silu(boundaries))
    result = torch.where(value < SILU_KNOTS[0], torch.zeros_like(value), result)
    return torch.where(value > SILU_KNOTS[-1], value, result)


def piecewise_exp(value: Tensor) -> Tensor:
    boundaries = value.new_tensor(EXP_KNOTS)
    result = _interpolate(value, boundaries, torch.exp(boundaries))
    result = torch.where(value < EXP_KNOTS[0], torch.zeros_like(value), result)
    return torch.where(value > EXP_KNOTS[-1], torch.full_like(value, math.e), result)
