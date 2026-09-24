import torch

from models.mamba.q_range_norm import QRangeNorm
from models.mamba.q_range_norm_math import round_ratio
from ptq.quant import QuantEntry, QuantProfile, QuantRuntime


def test_range_norm_integer_path_matches_scalar_oracle() -> None:
    profile = QuantProfile.from_entries({
        "norm.input": QuantEntry(0), "norm.gamma": QuantEntry(0),
        "norm.beta": QuantEntry(0), "norm.output": QuantEntry(0),
    })
    norm = QRangeNorm(
        torch.tensor([2.0, 2.0]), torch.tensor([1.0, 1.0]), 1e-6,
        "norm", QuantRuntime.frozen(profile),
    )
    assert norm(torch.tensor([[[1.0, 3.0]]])).tolist() == [[[0.0, 2.0]]]


def test_range_norm_constant_input_returns_quantized_beta() -> None:
    profile = QuantProfile.from_entries({
        name: QuantEntry(-2) for name in
        ("norm.input", "norm.gamma", "norm.beta", "norm.output")
    })
    norm = QRangeNorm(
        torch.ones(2), torch.tensor([0.25, -0.25]), 1e-6,
        "norm", QuantRuntime.frozen(profile),
    )
    assert norm(torch.ones(1, 1, 2)).tolist() == [[[0.25, -0.25]]]


def test_range_norm_retains_centered_term_when_range_is_below_epsilon() -> None:
    profile = QuantProfile.from_entries({
        "norm.input": QuantEntry(-20), "norm.gamma": QuantEntry(-7),
        "norm.beta": QuantEntry(-7), "norm.output": QuantEntry(-7),
    })
    norm = QRangeNorm(
        torch.ones(2), torch.zeros(2), 1e-6,
        "norm", QuantRuntime.frozen(profile),
    )
    result = norm(torch.tensor([[[0.0, 2.0 ** -20]]]))
    assert result.tolist() == [[[-0.4765625, 0.4765625]]]


def test_range_norm_cancels_tiny_epsilon_term_before_int64_widening() -> None:
    profile = QuantProfile.from_entries({
        "norm.input": QuantEntry(-126), "norm.gamma": QuantEntry(0),
        "norm.beta": QuantEntry(0), "norm.output": QuantEntry(0),
    })
    norm = QRangeNorm(
        torch.ones(2), torch.zeros(2), 1e-6,
        "norm", QuantRuntime.frozen(profile),
    )
    result = norm(torch.tensor([[[0.0, 2.0 ** -126]]]))
    assert result.tolist() == [[[0.0, 0.0]]]


def test_range_norm_constant_bypasses_oversized_epsilon_denominator() -> None:
    profile = QuantProfile.from_entries({
        "norm.input": QuantEntry(20), "norm.gamma": QuantEntry(-3),
        "norm.beta": QuantEntry(-4), "norm.output": QuantEntry(-5),
    })
    norm = QRangeNorm(
        torch.ones(2), torch.tensor([0.25, -0.25]), 1e-6,
        "norm", QuantRuntime.frozen(profile),
    )
    values = torch.tensor([[[1.0, 1.0]], [[0.0, 2.0 ** 20]]])
    result = norm(values)
    assert result[0].tolist() == [[0.25, -0.25]]
    assert result.shape == values.shape


def test_range_norm_handles_exact_int64_ratio_boundary() -> None:
    minimum = -(1 << 63)
    assert round_ratio(torch.tensor([minimum]), torch.tensor([2])).tolist() == [minimum // 2]
    profile = QuantProfile.from_entries({"norm.input": QuantEntry(0), "norm.gamma": QuantEntry(0), "norm.beta": QuantEntry(-54), "norm.output": QuantEntry(-54)})
    norm = QRangeNorm(torch.ones(20), torch.zeros(20), 1e-6, "norm", QuantRuntime.frozen(profile))
    values = torch.tensor([[[0.0, 26.0] + [27.0] * 18]])
    assert norm._integer_forward(values)[0, 0, 0].item() == -128.0 * 2.0 ** -54
    assert torch.isfinite(norm(values)).all()
