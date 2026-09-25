import hashlib
import json
from pathlib import Path

import pytest
import torch

import ptq.io as artifact_io
from models.emamba import EMamba
from models.q_emamba import QEMamba
from ptq.calibrate import ProfileObserver, candidate_profiles
from ptq.io import ArtifactError, load_quantized, save_quantized
from models.piecewise import PIECEWISE_SPEC
from ptq.quant import QuantRuntime
from ptq_emamba import parse_args
from training.checkpoint import MODEL_DEFAULTS


@pytest.mark.parametrize("use_pwl", [False, True])
def test_artifact_restores_nonlinear_mode_and_exact_output(tmp_path: Path, use_pwl: bool) -> None:
    torch.manual_seed(17)
    source, frames = EMamba().eval(), torch.randn(2, 8, 8, 5)
    observer = ProfileObserver()
    with torch.inference_mode():
        QEMamba(source, QuantRuntime.profiling(observer, use_pwl=use_pwl))(frames)
    profile = candidate_profiles(observer.snapshot())["max"]
    model = QEMamba(source, QuantRuntime.frozen(profile, use_pwl=use_pwl)).eval()
    hashes = {f"{split}.{kind}": "4" * 64 for split in ("train", "validation", "test")
              for kind in ("featuremap", "labels")}
    provenance = {
        "source_sha256": "1" * 64,
        "dataset_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "dataset_hashes": json.dumps(hashes, sort_keys=True),
        "indices_sha256": hashlib.sha256(b"[0,1]").hexdigest(),
        "calibration_indices": "[0,1]", "sample_count": 2, "seed": 0,
    }
    path = tmp_path / "quantized.pt"
    save_quantized(path, source, profile, {**MODEL_DEFAULTS, "readout": "flatten"},
                   provenance, use_pwl=use_pwl)
    loaded = load_quantized(path)
    assert loaded.model.runtime.use_pwl == use_pwl
    with torch.inference_mode():
        torch.testing.assert_close(loaded.model(frames), model(frames), rtol=0, atol=0)
    policy = loaded.metadata["quantization"]
    assert policy["nonlinear"] == ("piecewise_fp32" if use_pwl else "native_fp32")
    if use_pwl:
        assert policy["piecewise"] == PIECEWISE_SPEC
        raw = torch.load(path, weights_only=True)
        raw["metadata"]["quantization"]["piecewise"]["exp_knots"][0] = -5.0
        raw["sha256"] = artifact_io._digest(raw["metadata"], raw["parameters"])
        torch.save(raw, path)
        with pytest.raises(ArtifactError, match="numeric policy"):
            load_quantized(path)


def test_piecewise_cli_is_explicit_and_cannot_override_saved_mode() -> None:
    args = parse_args(["--checkpoint", "best.pt", "--output-dir", "new", "--piecewise"])
    assert args.piecewise is True
    assert parse_args(["--artifact", "quantized.pt"]).piecewise is None
    assert parse_args(["--checkpoint", "best.pt", "--output-dir", "new"]).piecewise is None
    assert parse_args(["--checkpoint", "best.pt", "--output-dir", "new", "--native"]).piecewise is False
    assert parse_args(["--checkpoint", "best.pt", "--output-dir", "new",
                       "--legacy-nonlinear", "native_fp32"]).legacy_nonlinear == "native_fp32"
    with pytest.raises(SystemExit):
        parse_args(["--artifact", "quantized.pt", "--piecewise"])
    with pytest.raises(SystemExit):
        parse_args(["--artifact", "quantized.pt", "--native"])
    with pytest.raises(SystemExit):
        parse_args(["--artifact", "quantized.pt", "--legacy-nonlinear", "native_fp32"])
