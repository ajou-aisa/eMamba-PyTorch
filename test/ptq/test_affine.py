from fractions import Fraction

import pytest
import torch
from torch import nn

from ptq.layers import QConv1d, QLinear
from ptq.quant import QuantEntry, QuantProfile, QuantRuntime


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def runtime_for(prefix: str, bias_exponent: int = -25) -> QuantRuntime:
    return QuantRuntime.frozen(QuantProfile.from_entries({
        "input": QuantEntry(0), "output": QuantEntry(1),
        f"{prefix}.input": QuantEntry(0),
        f"{prefix}.weight": QuantEntry(0),
        f"{prefix}.bias": QuantEntry(bias_exponent),
        "blocks.0.conv": QuantEntry(1),
    }))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("sign", [-1, 1])
def test_linear_keeps_bias_below_fp32_resolution(device: str, sign: int) -> None:
    # Given: a tiny bias moves an exact half-way result across the rounding tie.
    source = nn.Linear(1, 1, device=device)
    with torch.no_grad():
        source.weight.fill_(sign)
        source.bias.fill_(sign * 2.0 ** -25)
    layer = QLinear(source, "linear", "input", "output", runtime_for("linear"))
    # When: integer affine arithmetic precedes the final output rounding.
    result = layer(torch.ones(1, 1, device=device))
    # Then: exact arithmetic rounds beyond the tie; FP32 loses the tiny bias.
    assert result.dtype == torch.float32
    assert result.item() == sign * 2.0


@pytest.mark.parametrize("device", DEVICES)
def test_conv_keeps_bias_below_fp32_resolution(device: str) -> None:
    # Given: a two-channel depthwise convolution with opposite signed ties.
    source = nn.Conv1d(2, 2, 1, groups=2, device=device)
    assert source.bias is not None
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[[1.0]], [[-1.0]]], device=device))
        source.bias.copy_(torch.tensor([2.0 ** -25, -(2.0 ** -25)], device=device))
    runtime = runtime_for("blocks.0.conv1d")
    layer = QConv1d(source, "blocks.0.conv1d", runtime)
    # When: the existing block output boundary receives the convolution result.
    result = runtime.boundary("blocks.0.conv", layer(torch.ones(1, 2, 3, device=device)))
    # Then: both bias contributions survive integer accumulation.
    assert result.tolist() == [[[2.0] * 3, [-2.0] * 3]]


def test_bias_alignment_rejects_int64_overflow() -> None:
    # Given: aligning unit-scale products to a 2^-80 bias cannot fit INT64.
    source = nn.Linear(1, 1)
    with torch.no_grad():
        source.weight.fill_(1)
        source.bias.fill_(2.0 ** -80)
    layer = QLinear(source, "linear", "input", "output", runtime_for("linear", -80))
    # When/Then: unsupported alignment fails explicitly instead of wrapping.
    with pytest.raises(OverflowError, match="INT64"):
        layer(torch.ones(1, 1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_matches_exact_scaled_arithmetic(device: str, with_bias: bool) -> None:
    inputs = [[1, 2, -3, 4], [127, -128, 127, -128]]
    weights = [[-128, 127, -128, 127], [1, -2, 3, -4]]
    biases = [3, -5] if with_bias else [0, 0]
    runtime = QuantRuntime.frozen(QuantProfile.from_entries({
        "input": QuantEntry(-2), "output": QuantEntry(0),
        "linear.weight": QuantEntry(-4), "linear.bias": QuantEntry(-3),
    }))
    source = nn.Linear(4, 2, bias=with_bias, device=device)
    with torch.no_grad():
        source.weight.copy_(torch.tensor(weights, device=device) / 16)
        if source.bias is not None:
            source.bias.copy_(torch.tensor(biases, device=device) / 8)
    layer = QLinear(source, "linear", "input", "output", runtime)
    expected = [[max(-128, min(127, round(Fraction(
        sum(x * w for x, w in zip(row, weight, strict=True)) + 8 * bias, 64,
    )))) for weight, bias in zip(weights, biases, strict=True)] for row in inputs]
    values = torch.tensor(inputs, device=device).float().div(4).unsqueeze(0)
    result = layer(values)
    assert result.tolist() == [expected]
