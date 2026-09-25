import pytest
import torch
from torch import Tensor

from models.emamba import EMamba
from models.q_emamba import QEMamba
from ptq.calibrate import ProfileObserver, candidate_profiles
from ptq.piecewise import piecewise_exp, piecewise_silu
from ptq.quant import QuantRuntime


@pytest.mark.parametrize("use_pwl", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_gate_and_abar_use_selected_function_in_both_paths(
    use_pwl: bool, frozen: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    source, frames = EMamba().eval(), torch.randn(2, 8, 8, 5)
    observer = ProfileObserver()
    runtime = QuantRuntime.profiling(observer, use_pwl=use_pwl)
    with torch.inference_mode():
        QEMamba(source, runtime)(frames)
    if frozen:
        runtime = QuantRuntime.frozen(candidate_profiles(observer.snapshot())["max"], use_pwl=use_pwl)
    model = QEMamba(source, runtime).eval()
    captured: dict[str, list[tuple[Tensor, Tensor]]] = {}
    original = QuantRuntime.boundary

    def boundary(self: QuantRuntime, name: str, value: Tensor, bits: int = 8) -> Tensor:
        result = original(self, name, value, bits)
        captured.setdefault(name, []).append((value.detach().clone(), result.detach().clone()))
        return result

    monkeypatch.setattr(QuantRuntime, "boundary", boundary)
    with torch.inference_mode():
        model(frames)
    for index in range(2):
        prefix = f"blocks.{index}"
        gate_input = captured[f"{prefix}.gateProjection"][-1][1]
        expected_gate = piecewise_silu(gate_input) if use_pwl else torch.nn.functional.silu(gate_input)
        torch.testing.assert_close(captured[f"{prefix}.gate"][0][0], expected_gate, rtol=0, atol=0)
        arguments, outputs = captured[f"{prefix}.ssm.expInput"], captured[f"{prefix}.ssm.Abar"]
        assert len(arguments) == len(outputs) == 16
        for (_, argument), (output, _) in zip(arguments, outputs):
            expected = piecewise_exp(argument) if use_pwl else torch.exp(argument)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        delta_input = captured[f"{prefix}.ssm.deltaProjection"][-1][1]
        torch.testing.assert_close(captured[f"{prefix}.ssm.delta"][0][0], torch.relu(delta_input))
