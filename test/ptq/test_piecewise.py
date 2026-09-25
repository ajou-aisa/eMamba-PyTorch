import bisect
import math
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from ptq.piecewise import EXP_KNOTS, SILU_KNOTS, piecewise_exp, piecewise_silu


def reference(value: float, knots: tuple[float, ...], function: Callable[[float], float]) -> float:
    index = max(0, min(len(knots) - 2, bisect.bisect_left(knots, value) - 1))
    left, right = knots[index:index + 2]
    return function(left) + (value - left) * (function(right) - function(left)) / (right - left)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["silu", "exp"])
def test_piecewise_segments_and_tails_match_scalar_reference(device: str, kind: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    knots = SILU_KNOTS if kind == "silu" else EXP_KNOTS
    function = (lambda x: x / (1 + math.exp(-x))) if kind == "silu" else math.exp
    approximation = piecewise_silu if kind == "silu" else piecewise_exp
    points = sorted({k + offset for k in knots for offset in (-1e-4, 0, 1e-4)}
                    | {(a + b) / 2 for a, b in zip(knots, knots[1:])} | {-20.0, 20.0})
    values = torch.tensor(points, dtype=torch.float32, device=device)
    expected = []
    for value in values.cpu().tolist():
        result = reference(value, knots, function)
        if value < knots[0]:
            result = 0.0
        if value > knots[-1]:
            result = value if kind == "silu" else math.e
        expected.append(result)
    output = approximation(values)
    torch.testing.assert_close(output, values.new_tensor(expected), rtol=2e-6, atol=2e-7)
    assert output.dtype == values.dtype and output.device == values.device


def test_piecewise_evaluates_linear_segments_instead_of_native_curve() -> None:
    value = torch.tensor([0.25])
    expected_silu = torch.nn.functional.silu(torch.tensor([0.5])) / 2
    torch.testing.assert_close(piecewise_silu(value), expected_silu)
    assert not torch.equal(piecewise_silu(value), torch.nn.functional.silu(value))
    argument = torch.tensor([-0.25])
    expected_exp = (torch.exp(torch.tensor([-0.5])) + 1) / 2
    torch.testing.assert_close(piecewise_exp(argument), expected_exp)
    assert not torch.equal(piecewise_exp(argument), torch.exp(argument))


@pytest.mark.parametrize("approximation", [piecewise_silu, piecewise_exp])
def test_piecewise_accepts_noncontiguous_input(approximation: Callable[[Tensor], Tensor]) -> None:
    values = torch.linspace(-8, 8, 24).reshape(2, 3, 4).transpose(1, 2)
    torch.testing.assert_close(approximation(values), approximation(values.contiguous()), rtol=0, atol=0)
