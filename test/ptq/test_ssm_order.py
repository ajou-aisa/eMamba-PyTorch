import pytest
import torch

from models.emamba import EMamba
from models.q_emamba import QEMamba
from ptq.calibrate import ProfileObserver, candidate_profiles
from ptq.quant import QuantRuntime


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_discretization_precedes_each_token_state_update(
    device: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a calibrated model and a recorder that preserves real quantization.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(12)
    source = EMamba().to(device).eval()
    frames = torch.randn(1, 8, 8, 5, device=device)
    observer = ProfileObserver()
    with torch.inference_mode():
        QEMamba(source, QuantRuntime.profiling(observer))(frames)
    profile = candidate_profiles(observer.snapshot())["max"]
    model = QEMamba(source, QuantRuntime.frozen(profile)).eval()
    stages = ("expInput", "Abar", "Bbar", "currentState", "y.preclip", "state")
    events: list[tuple[str, tuple[int, ...]]] = []
    original = QuantRuntime.record_codes

    def record(runtime: QuantRuntime, name: str, codes: torch.Tensor, bits: int) -> None:
        if name.startswith("blocks.0.ssm."):
            stage = name.removeprefix("blocks.0.ssm.")
            if stage in stages:
                events.append((stage, tuple(codes.shape)))
        original(runtime, name, codes, bits)

    monkeypatch.setattr(QuantRuntime, "record_codes", record)
    # When frozen inference processes the sixteen tokens.
    with torch.inference_mode():
        output = model(frames)
    # Then each token is discretized before its recurrence and state storage.
    assert [stage for stage, _ in events] == list(stages) * 16
    assert all(len(shape) == 3 for stage, shape in events if stage in stages[:3])
    assert output.shape == (1, 57)
    assert torch.isfinite(output).all()
