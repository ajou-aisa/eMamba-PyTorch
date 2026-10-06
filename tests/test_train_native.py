import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import train
from models.emamba import EMamba
from models.mamba import block, selective_ssm
from training import session
from training.checkpoint import MODEL_DEFAULTS, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("mode", ["train", "smoke"])
def test_training_cli_uses_native_nonlinear_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    frames = torch.linspace(-1, 1, 640).reshape(2, 8, 8, 5)
    loader = DataLoader(TensorDataset(frames, torch.zeros(2, 57)), batch_size=2)
    monkeypatch.setattr(session, "build_dataloaders", lambda *_args: {
        "train": loader, "validation": loader,
    })
    monkeypatch.setattr(session, "training_data_fingerprint", lambda _path: "a" * 64)

    def reject_piecewise(_value: torch.Tensor) -> torch.Tensor:
        pytest.fail("FP training invoked a piecewise nonlinear function")

    monkeypatch.setattr(block, "piecewise_silu", reject_piecewise)
    monkeypatch.setattr(selective_ssm, "piecewise_exp", reject_piecewise)
    output_dir = tmp_path / "run"
    length_option = "--epochs" if mode == "train" else "--steps"
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--mode", mode, "--device", "cpu", length_option, "1",
        "--output-dir", str(output_dir), "--no-progress",
    ])

    train.main()

    if mode == "train":
        restored, payload = load_checkpoint(output_dir / "last.pt", torch.device("cpu"))
        assert payload["nonlinear_policy"] == "native_fp32"
        assert torch.isfinite(restored(frames)).all()
    else:
        assert (output_dir / "smoke.json").is_file()


def test_training_cli_rejects_piecewise_resume_without_changing_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = EMamba(nonlinear_policy="piecewise_fp32")
    checkpoint = tmp_path / "last.pt"
    save_checkpoint(
        checkpoint, model, torch.optim.AdamW(model.parameters()), 1, 1, 1.0,
        {**MODEL_DEFAULTS, "readout": "flatten"}, {"seed": 0}, torch.device("cpu"),
    )
    original = checkpoint.read_bytes()
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--mode", "train", "--resume", str(checkpoint),
        "--epochs", "2", "--device", "cpu", "--no-progress",
    ])

    with pytest.raises(ValueError, match="native_fp32"):
        train.main()

    assert checkpoint.read_bytes() == original
