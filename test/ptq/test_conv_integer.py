from fractions import Fraction

import pytest
import torch
from torch import nn

from models.mamba.block import MambaConv1D
from ptq.layers import QConv1d
from ptq.quant import QuantEntry, QuantProfile, QuantRuntime


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_depthwise_conv_preserves_causal_padding_and_channel_scales(
    device: str, with_bias: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    inputs = [[127, -128, 3, -5, 0], [-128, 127, -7, 9, 1]]
    weights = [[127, -128, 3, 2], [-128, 127, -1, 4]]
    biases = [3, -5] if with_bias else [0, 0]
    runtime = QuantRuntime.frozen(QuantProfile.from_entries({
        "blocks.0.conv1d.input": QuantEntry(-3),
        "blocks.0.conv1d.weight": QuantEntry(-4),
        "blocks.0.conv1d.bias": QuantEntry(-6),
        "blocks.0.conv": QuantEntry(-3),
    }))
    source = nn.Conv1d(2, 2, 4, padding=3, groups=2, bias=with_bias, device=device)
    with torch.no_grad():
        source.weight.copy_(torch.tensor(weights, device=device).unsqueeze(1) / 16)
        if source.bias is not None:
            source.bias.copy_(torch.tensor(biases, device=device) / 64)
    wrapper = MambaConv1D(2)
    wrapper.add_module("conv1d", QConv1d(source, "blocks.0.conv1d", runtime))
    expected = []
    for row, kernel, bias in zip(inputs, weights, biases, strict=True):
        padded = [0, 0, 0] + row
        channel = []
        for time in range(len(row)):
            accumulator = sum(padded[time + tap] * kernel[tap] for tap in range(4))
            code = round(Fraction(accumulator + 2 * bias, 16))
            channel.append(max(-128, min(127, code)) / 8)
        expected.append(channel)
    values = torch.tensor(inputs, device=device).float().div(8).T.unsqueeze(0)
    result = runtime.boundary("blocks.0.conv", wrapper(values))
    expected_tensor = torch.tensor(expected, device=device).T.unsqueeze(0)
    torch.testing.assert_close(result, expected_tensor, rtol=0, atol=0)
