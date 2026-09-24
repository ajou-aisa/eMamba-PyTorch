from functools import lru_cache

import torch
from brevitas import nn as qnn
from brevitas.inject.enum import ScalingImplType
from brevitas.quant import Int8ActPerTensorFixedPoint, Int8WeightPerTensorFixedPoint


class FixedAct(Int8ActPerTensorFixedPoint):
    scaling_impl_type = ScalingImplType.CONST
    narrow_range = False


class FixedWeight(Int8WeightPerTensorFixedPoint):
    scaling_impl_type = ScalingImplType.CONST
    narrow_range = False


@lru_cache(maxsize=None)
def fixed_identity(exponent: int, bits: int, device: torch.device) -> qnn.QuantIdentity:
    scale, magnitude = 2.0 ** exponent, 1 << (bits - 1)
    return qnn.QuantIdentity(
        act_quant=FixedAct, bit_width=bits, signed=True, narrow_range=False,
        scaling_const=scale * magnitude, min_val=-scale * magnitude,
        max_val=scale * (magnitude - 1), return_quant_tensor=False,
    ).to(device).eval()


def fixed_weight_arguments(exponent: int) -> dict[str, float | bool | type]:
    return {
        "weight_quant": FixedWeight,
        "weight_scaling_const": (2.0 ** exponent) * 128,
        "weight_narrow_range": False,
    }
