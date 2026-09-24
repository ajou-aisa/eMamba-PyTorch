import torch.nn.functional as functional
from torch import Tensor, nn

from .ops import INT64_MAX, align_codes, checked_add, dequantize_codes, saturate
from .quant import QuantizationError, QuantRuntime


def sum_products(left: Tensor, right: Tensor) -> Tensor:
    bound = left.shape[-1] * int(left.abs().max()) * int(right.abs().max())
    if bound > INT64_MAX:
        raise OverflowError("affine accumulation would overflow INT64")
    return (left * right).sum(dim=-1)


def linear_accumulator(inputs: Tensor, weights: Tensor) -> Tensor:
    return sum_products(inputs.unsqueeze(-2), weights)


def conv_accumulator(inputs: Tensor, weights: Tensor, layer: nn.Conv1d) -> Tensor:
    if layer.groups != layer.in_channels or layer.out_channels != layer.in_channels:
        raise QuantizationError("integer Conv1d requires depthwise channels")
    if layer.padding_mode != "zeros" or isinstance(layer.padding, str):
        raise QuantizationError("integer Conv1d requires numeric zero padding")
    padding = layer.padding[0]
    padded = functional.pad(inputs, (padding, padding))
    dilation = layer.dilation[0]
    width = (layer.kernel_size[0] - 1) * dilation + 1
    windows = padded.unfold(-1, width, layer.stride[0])[..., ::dilation]
    return sum_products(windows, weights[:, 0].unsqueeze(1))


def affine_output(
    accumulator: Tensor, bias: Tensor | None, runtime: QuantRuntime, *,
    prefix: str, input_name: str, output_name: str,
) -> Tensor:
    exponent = runtime.entry(input_name).exponent + runtime.entry(f"{prefix}.weight").exponent
    if bias is not None:
        bias_name = f"{prefix}.bias"
        bias_exponent = runtime.entry(bias_name).exponent
        bias_codes = runtime.codes(bias_name, bias)
        common = min(exponent, bias_exponent)
        accumulator = checked_add(
            align_codes(accumulator, exponent, common),
            align_codes(bias_codes, bias_exponent, common),
        )
        exponent = common
    output_exponent = runtime.entry(output_name).exponent
    codes = align_codes(accumulator, exponent, output_exponent)
    runtime.record_codes(f"{output_name}.preclip", codes, 8)
    return dequantize_codes(saturate(codes, 8), output_exponent)
