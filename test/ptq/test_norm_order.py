from fractions import Fraction

import pytest
import torch

from models.mamba.q_range_norm import QRangeNorm
from models.mamba.range_norm import RangeNorm
from ptq.quant import QuantEntry, QuantProfile, QuantRuntime


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_integer_norm_divides_before_gamma_and_beta(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    unit = 1 << 24
    inputs = [0, 1, 1]
    mean = round(Fraction(sum(inputs) * unit, len(inputs)))
    centered = [x * unit - mean for x in inputs]
    span = max(centered) - min(centered)
    normalized = [round(Fraction(x * unit, span)) for x in centered]
    expected = [round(Fraction(x * 3 + unit, 2 * unit)) * 2 for x in normalized]
    profile = QuantProfile.from_entries({
        "norm.input": QuantEntry(0), "norm.gamma": QuantEntry(0),
        "norm.beta": QuantEntry(0), "norm.output": QuantEntry(1),
    })
    norm = QRangeNorm(torch.full((3,), 3.0, device=device), torch.ones(3, device=device),
                      1e-6, "norm", QuantRuntime.frozen(profile))
    result = norm(torch.tensor([[inputs]], dtype=torch.float32, device=device))
    assert result.tolist() == [[expected]]


def test_calibration_uses_original_float_operation_order() -> None:
    torch.manual_seed(73)
    source = RangeNorm(20)
    with torch.no_grad():
        source.gamma.copy_(torch.linspace(0.1, 2.0, 20))
    tokens = torch.randn(2, 16, 20)
    norm = QRangeNorm(source.gamma, source.beta, source.eps, "norm", QuantRuntime.bypass())
    torch.testing.assert_close(norm(tokens), source(tokens), rtol=0, atol=0)
