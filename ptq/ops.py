import torch
from collections.abc import Callable
from torch import Tensor
from .checked import INT64_MAX, checked_add, checked_mul
class ArithmeticConfigurationError(ValueError):
    pass
def signed_limits(bits: int) -> tuple[int, int]:
    if bits < 2 or bits > 63:
        raise ArithmeticConfigurationError("bits must be between 2 and 63")
    return -(1 << (bits - 1)), (1 << (bits - 1)) - 1
def saturate(value: Tensor, bits: int) -> Tensor:
    lower, upper = signed_limits(bits)
    return value.clamp(lower, upper)
def quantize_codes(value: Tensor, exponent: int, bits: int = 8) -> Tensor:
    if type(exponent) is not int or not -126 <= exponent <= 127:
        raise ArithmeticConfigurationError("exponent must be between -126 and 127")
    scale = 2.0 ** exponent
    return saturate(torch.round(value.detach().to(torch.float64) / scale), bits).to(torch.int64)
def dequantize_codes(codes: Tensor, exponent: int, dtype: torch.dtype = torch.float32) -> Tensor:
    integer_codes = codes.to(torch.int64)
    if ((integer_codes > (1 << 53)) | (integer_codes < -(1 << 53))).any():
        raise OverflowError("integer-to-float bridge exceeds exact FP64 range")
    return integer_codes.to(torch.float64).mul_(2.0 ** exponent).to(dtype)
def round_shift(value: Tensor, shift: int) -> Tensor:
    if shift < 0:
        width = -shift
        if width >= 63:
            if width == 63 and ((value == 0) | (value == -1)).all():
                return value << width
            raise OverflowError("integer left shift would overflow INT64")
        if ((value > (INT64_MAX >> width)) | (value < (-(1 << 63) >> width))).any():
            raise OverflowError("integer left shift would overflow INT64")
        return value << width
    if shift == 0:
        return value.clone()
    if shift >= 63:
        raise OverflowError("integer right shift exceeds checked INT64 domain")
    quotient = value >> shift
    remainder = value - (quotient << shift)
    half = 1 << (shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    return quotient + increment.to(torch.int64)
def align_codes(value: Tensor, source_exponent: int, target_exponent: int) -> Tensor:
    return round_shift(value, target_exponent - source_exponent)
def integer_ssm_step(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor, state: Tensor, *, x_exponent: int,
    b_exponent: int, c_exponent: int, d_exponent: int, state_exponent: int,
    output_exponent: int, recorder: Callable[[str, Tensor, int], None] | None = None,
    prefix: str = "ssm",
) -> tuple[Tensor, Tensor, Tensor]:
    current_exponent = state_exponent - 7
    recurrent = checked_mul(a, state)
    injected = checked_mul(b, x.unsqueeze(-1))
    injected = align_codes(injected, b_exponent + x_exponent, current_exponent)
    raw_current = checked_add(recurrent, injected)
    if recorder is not None:
        recorder(f"{prefix}.currentState", raw_current, 24)
    current = saturate(raw_current, 24)
    state_products = checked_mul(current, c.unsqueeze(1))
    reduction_bound = int(state_products.abs().max()) * state_products.shape[-1]
    if reduction_bound > INT64_MAX:
        raise OverflowError("state reduction would overflow INT64")
    state_path = state_products.sum(-1)
    direct = checked_mul(x, d)
    common = min(current_exponent + c_exponent, x_exponent + d_exponent)
    accumulator = checked_add(
        align_codes(state_path, current_exponent + c_exponent, common),
        align_codes(direct, x_exponent + d_exponent, common),
    )
    raw_output = align_codes(accumulator, common, output_exponent)
    if recorder is not None:
        recorder(f"{prefix}.y.preclip", raw_output, 8)
    output = saturate(raw_output, 8)
    raw_state = current >> 7
    if recorder is not None:
        recorder(f"{prefix}.state", raw_state, 17)
    return output, current, saturate(raw_state, 17)
def integer_ssm(
    x: Tensor, a: Tensor, b: Tensor, c: Tensor, d: Tensor, *, x_exponent: int,
    b_exponent: int, c_exponent: int, d_exponent: int, state_exponent: int,
    output_exponent: int, recorder: Callable[[str, Tensor, int], None] | None = None,
    prefix: str = "ssm",
) -> tuple[Tensor, Tensor, Tensor]:
    state = torch.zeros_like(a[:, 0], dtype=torch.int64)
    outputs, currents, stored = [], [], []
    for step in range(x.shape[1]):
        output, current, state = integer_ssm_step(
            x[:, step], a[:, step], b[:, step], c[:, step], d, state,
            x_exponent=x_exponent, b_exponent=b_exponent, c_exponent=c_exponent,
            d_exponent=d_exponent, state_exponent=state_exponent,
            output_exponent=output_exponent, recorder=recorder, prefix=prefix,
        )
        outputs.append(output)
        currents.append(current)
        stored.append(state)
    return torch.stack(outputs, 1), torch.stack(currents, 1), torch.stack(stored, 1)
