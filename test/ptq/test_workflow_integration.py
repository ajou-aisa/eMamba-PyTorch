from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import ptq.workflow as workflow
from models.emamba import EMamba, NonlinearPolicy
from models.q_emamba import QEMamba
from ptq.calibrate import ProfileObserver, candidate_profiles
from ptq.io import load_quantized
from ptq.model import prepare_model
from ptq.quant import QuantRuntime
from ptq.report import JsonValue
from training.checkpoint import MODEL_DEFAULTS


@pytest.mark.parametrize(("policy", "use_pwl", "expected"), [
    ("native_fp32", None, "native_fp32"),
    ("piecewise_fp32", None, "piecewise_fp32"),
    ("native_fp32", True, "piecewise_fp32"),
    ("piecewise_fp32", False, "native_fp32"),
])
def test_conversion_artifact_roundtrip_accepts_full_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: NonlinearPolicy,
    use_pwl: bool | None, expected: NonlinearPolicy,
) -> None:
    # Given: one real model/profile and tiny distinct split fixtures.
    torch.manual_seed(23)
    source = EMamba(nonlinear_policy=policy)
    frames = torch.randn(3, 8, 8, 5)
    base = TensorDataset(frames, torch.zeros(3, 57))
    datasets = {name: Subset(base, [index]) for index, name in enumerate(
        ("train", "validation", "test"))}
    loaders = {name: DataLoader(dataset, batch_size=1) for name, dataset in datasets.items()}
    observer = ProfileObserver()
    prepare_model(source, QuantRuntime.profiling(observer))(frames[:1])
    profile = candidate_profiles(observer.snapshot())["max"]
    config = {**MODEL_DEFAULTS, "readout": "flatten"}
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    for suffix in ("train", "validate", "test"):
        for kind in ("featuremap", "labels"):
            (tmp_path / f"{kind}_{suffix}.npy").write_bytes(f"{kind}-{suffix}".encode())

    def metric(_model: torch.nn.Module, loader: DataLoader, _device: torch.device,
               split: str, _checkpoint: str) -> Mapping[str, JsonValue]:
        return {"split": split, "samples": len(loader.dataset), "rmse_cm": {"all": 1.0}}

    monkeypatch.setattr(workflow, "build_dataloaders", lambda *_args: loaders)
    monkeypatch.setattr(workflow, "load_checkpoint", lambda *_args, **_kwargs: (source, {"model_config": config}))
    monkeypatch.setattr(workflow, "collect_calibration", lambda *_args: None)
    monkeypatch.setattr(workflow, "candidate_profiles", lambda _snapshot: {"max": profile, "percentile": profile})
    monkeypatch.setattr(workflow, "evaluate", metric)
    # When: workflow conversion uses the real artifact writer and loader.
    result = workflow.convert(checkpoint, tmp_path / "result", tmp_path, "test",
                              torch.device("cpu"), batch_size=1, calibration_count=1,
                              use_pwl=use_pwl)
    loaded = load_quantized(tmp_path / "result/quantized.pt")
    # Then: strict reload accepts the complete reproducibility provenance.
    assert result["reload"]["output_codes_equal"] is True
    assert type(loaded.model) is QEMamba
    assert result["reload"]["frames_compared"] == 1
    assert {"dataset_hashes", "calibration_indices"} <= set(loaded.metadata["provenance"])
    assert loaded.metadata["quantization"]["nonlinear"] == expected
