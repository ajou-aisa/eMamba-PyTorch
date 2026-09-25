from typing import Final

import torch
from torch import Tensor

from ptq.ops import align_codes

NORM_FRACTION_BITS: Final = 24


def left_shift_checked(value: Tensor, width: int) -> Tensor:
    return align_codes(value, 0, -width)


def round_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    quotient = torch.div(numerator, denominator, rounding_mode="floor")
    remainder = numerator - quotient * denominator
    half = denominator >> 1
    increment = (remainder > half) | (
        (remainder == half) & ((denominator & 1) == 0) & ((quotient & 1) == 1)
    )
    return quotient + increment.to(torch.int64)
