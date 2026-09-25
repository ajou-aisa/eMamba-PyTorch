from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

import ptq_emamba
import ptq.workflow as workflow
from ptq.quant import QuantEntry, QuantProfile, QuantRuntime
from ptq.report import JsonValue
from training.checkpoint import MODEL_DEFAULTS, save_checkpoint
from models.emamba import EMamba


def test_artifact_mode_defaults_to_validation_and_auto_device() -> None:
    # Given: the smallest source-free reload invocation.
    argv = ["--artifact", "quantized.pt"]
    # When: CLI arguments cross the parser boundary.
    arguments = ptq_emamba.parse_args(argv)
    # Then: reload follows the established validation and auto-device defaults.
    assert arguments.artifact == Path("quantized.pt")
    assert arguments.checkpoint is None
    assert arguments.split == "validation"
    assert arguments.device == "auto"


def test_conversion_requires_a_fresh_output_directory(tmp_path: Path) -> None:
    # Given: conversion without its required destination.
    argv = ["--checkpoint", str(tmp_path / "best.pt")]
    # When/Then: argument parsing rejects the incomplete workflow.
    with pytest.raises(SystemExit):
        ptq_emamba.parse_args(argv)


def test_missing_checkpoint_does_not_reserve_output_directory(tmp_path: Path) -> None:
    # Given: a missing checkpoint and fresh output path.
    path = tmp_path / "best.pt"
    output = tmp_path / "conversion"
    # When: conversion rejects the checkpoint.
    with pytest.raises(FileNotFoundError):
        workflow.convert(path, output, tmp_path, "validation", torch.device("cpu"))
    # Then: the output path remains available.
    assert not output.exists()


def test_missing_data_does_not_reserve_output_directory(tmp_path: Path) -> None:
    # Given: a valid checkpoint and missing data.
    path = tmp_path / "best.pt"
    output = tmp_path / "conversion"
    model = EMamba()
    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 2, 1.0,
                    {**MODEL_DEFAULTS, "readout": "flatten"}, {"seed": 0}, torch.device("cpu"))
    # When: the dataset boundary rejects conversion.
    with pytest.raises(ValueError, match="feature file"):
        workflow.convert(path, output, tmp_path, "validation", torch.device("cpu"))
    # Then: the output path remains available.
    assert not output.exists()


class FixtureModel(nn.Module):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name
        self.nonlinear_policy = "piecewise_fp32"

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.zeros(frames.shape[0], 57, device=frames.device)


@pytest.mark.parametrize("requested", ["validation", "test"])
def test_conversion_selects_on_validation_and_calibrates_on_train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    requested: Literal["validation", "test"],
) -> None:
    import ptq.workflow as workflow

    # Given: distinct fixture loaders and candidate validation scores.
    base = TensorDataset(torch.arange(6.0).unsqueeze(1), torch.zeros(6, 57))
    datasets = {
        "train": Subset(base, [0, 1]), "validation": Subset(base, [2, 3]),
        "test": Subset(base, [4, 5]),
    }
    loaders = {name: DataLoader(dataset, batch_size=1) for name, dataset in datasets.items()}
    profile = QuantProfile.from_entries({"output": QuantEntry(0)})
    events: list[tuple[str, str]] = []

    def prepare(_source: nn.Module, runtime: QuantRuntime) -> FixtureModel:
        selected = getattr(runtime, "profile", None)
        return FixtureModel("profile" if selected is None else "max")

    def evaluate(model: FixtureModel, loader: DataLoader, _device: torch.device,
                 split: str, _checkpoint: str, **_options: bool) -> Mapping[str, JsonValue]:
        events.append((model.name, split))
        return {"split": split, "samples": len(loader.dataset),
                "rmse_cm": {"all": 1.0 if model.name == "max" else 2.0}}

    monkeypatch.setattr(workflow, "build_dataloaders", lambda *_args, **_kwargs: loaders)
    monkeypatch.setattr(workflow, "load_checkpoint", lambda *_args, **_kwargs: (FixtureModel("fp32"), {"model_config": {}}))
    monkeypatch.setattr(workflow, "QEMamba", prepare)
    monkeypatch.setattr(workflow, "evaluate", evaluate)
    monkeypatch.setattr(workflow, "candidate_profiles", lambda _snapshot: {"max": profile, "percentile": profile})
    monkeypatch.setattr(workflow, "collect_calibration", lambda _model, dataset, *_args, **_kwargs: events.append(("calibrate", next(name for name, item in datasets.items() if item is dataset))))
    monkeypatch.setattr(workflow, "save_quantized", lambda path, *_args: path.write_bytes(b"artifact") or "digest")
    monkeypatch.setattr(workflow, "load_quantized", lambda _path: SimpleNamespace(model=FixtureModel("max"), profile=profile))
    monkeypatch.setattr(workflow, "file_sha256", lambda _path: "sha256")
    # When: conversion targets either supported reporting split.
    (tmp_path / "best.pt").touch()
    workflow.convert(tmp_path / "best.pt", tmp_path / requested, tmp_path, requested,
                     torch.device("cpu"), batch_size=1, calibration_count=1)
    # Then: calibration is train-only and both candidates are selected on validation.
    assert ("calibrate", "train") in events
    assert events.count(("max", "validation")) >= 2
    assert not any(name == "calibrate" and split == "test" for name, split in events)
