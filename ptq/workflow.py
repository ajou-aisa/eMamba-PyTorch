import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Literal
import torch
from torch.utils.data import DataLoader
from models.emamba import NonlinearPolicy
from ptq.calibrate import (ProfileObserver, candidate_profiles, collect_calibration,
                           index_hash, profile_hash, select_indices)
from ptq.io import ArtifactError, load_quantized, save_quantized
from ptq.model import QEMamba
from ptq.quant import QuantProfile, QuantRuntime
from ptq.report import WorkflowReport
from training.checkpoint import load_checkpoint
from datasets.mars import SPLIT_FILES, build_dataloaders
from training.engine import evaluate
from training.runtime import configure_fp32

Split = Literal["validation", "test"]
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _data_hashes(data_root: Path) -> dict[str, str]:
    return {
        f"{split}.{kind}": file_sha256(data_root / f"{kind}_{suffix}.npy")
        for split, suffix in SPLIT_FILES.items() for kind in ("featuremap", "labels")
    }
def _codes_equal(before: torch.nn.Module, after: torch.nn.Module, loader: DataLoader,
                 device: torch.device, profile: QuantProfile) -> tuple[bool, int]:
    runtime = QuantRuntime.frozen(profile)
    after.to(device).eval()
    compared = 0
    with torch.inference_mode():
        for features, _labels in loader:
            frames = features.to(device)
            left = runtime.codes("output", before(frames))
            right = runtime.codes("output", after(frames))
            if not torch.equal(left, right):
                return False, compared
            compared += frames.shape[0]
    return True, compared
def convert(checkpoint: Path, output_dir: Path, data_root: Path, split: Split,
            device: torch.device, *, batch_size: int = 64,
            calibration_count: int = 2_048, use_pwl: bool | None = None,
            legacy_nonlinear: NonlinearPolicy | None = None,
            precision: dict[str, str] | None = None) -> WorkflowReport:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    precision = configure_fp32(device) if precision is None else precision
    requested = ("train", "validation") if split == "validation" else (
        "train", "validation", "test")
    loaders = build_dataloaders(data_root, requested, batch_size, 0)
    source, payload = load_checkpoint(checkpoint, device, legacy_nonlinear=legacy_nonlinear)
    use_pwl = source.nonlinear_policy == "piecewise_fp32" if use_pwl is None else use_pwl
    baseline = evaluate(source, loaders[split], device, split, str(checkpoint))
    observer = ProfileObserver()
    profiling = QEMamba(source, QuantRuntime.profiling(observer, use_pwl=use_pwl)).to(device)
    indices = select_indices(len(loaders["train"].dataset), calibration_count, 0)
    collect_calibration(profiling, loaders["train"].dataset, indices, device, batch_size)
    candidates = candidate_profiles(observer.snapshot())
    candidate_metrics, candidate_hashes = {}, {}
    selected, best = "max", float("inf")
    for name in ("max", "percentile"):
        runtime = QuantRuntime.frozen(candidates[name], use_pwl=use_pwl)
        model = QEMamba(source, runtime).to(device)
        metric = evaluate(model, loaders["validation"], device, "validation", str(checkpoint))
        candidate_metrics[name], candidate_hashes[name] = metric, profile_hash(candidates[name])
        score = metric["rmse_cm"]["all"]
        if score < best:
            selected, best = name, score
    profile = candidates[selected]
    runtime = QuantRuntime.frozen(profile, use_pwl=use_pwl)
    model = QEMamba(source, runtime).to(device)
    quantized = evaluate(model, loaders[split], device, split, str(checkpoint))
    statistics = {name: asdict(item) for name, item in runtime.statistics().items()}
    train_dataset = loaders["train"].dataset
    source_indices = tuple(int(train_dataset.indices[index]) for index in indices)
    hashes = _data_hashes(data_root)
    provenance = {"source_sha256": file_sha256(checkpoint), "dataset_sha256": hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()).hexdigest(), "dataset_hashes": json.dumps(hashes,
        sort_keys=True), "indices_sha256": index_hash(source_indices),
        "calibration_indices": json.dumps(source_indices), "sample_count": len(indices), "seed": 0}
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "quantized.pt"
    save_quantized(artifact_path, source, profile, payload["model_config"], provenance, use_pwl)
    loaded = load_quantized(artifact_path)
    equal, compared = _codes_equal(model, loaded.model, loaders[split], device, loaded.profile)
    if not equal:
        raise ArtifactError("reloaded output codes differ")
    report = {"schema_version": 1, "selection": {"split": "validation", "selected": selected,
              "candidates": candidate_metrics, "profile_hashes": candidate_hashes},
              "evaluation": {"split": split, "samples": quantized["samples"],
              "fp32": baseline, "ptq": quantized, "precision": precision}, "calibration": provenance,
              "quantization": {"logical_widths": {name: entry.bits for name, entry in profile.entries.items()},
              "statistics": statistics}, "artifact": {"path": "quantized.pt",
              "sha256": file_sha256(artifact_path)}, "reload": {"output_codes_equal": equal,
              "frames_compared": compared, "profile_hash": profile_hash(loaded.profile)}}
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
