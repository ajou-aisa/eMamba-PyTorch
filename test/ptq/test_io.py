import hashlib
import json
from pathlib import Path

import pytest
import torch

import ptq.io as artifact_io
import ptq.parameters as parameter_io
from models.emamba import EMamba
from ptq.calibrate import ProfileObserver, candidate_profiles
from ptq.io import ArtifactError, load_quantized, save_quantized
from ptq.model import prepare_model
from ptq.quant import QuantRuntime
from training.checkpoint import MODEL_DEFAULTS


def _fixture(path: Path) -> tuple[torch.Tensor, EMamba]:
    torch.manual_seed(17)
    source = EMamba()
    source.blocks[0].norm.eps = 1.25e-5
    frames = torch.randn(2, 8, 8, 5)
    observer = ProfileObserver()
    prepare_model(source, QuantRuntime.profiling(observer))(frames)
    profile = candidate_profiles(observer.snapshot())["max"]
    expected = prepare_model(source, QuantRuntime.frozen(profile))
    hashes = {f"{split}.{kind}": "4" * 64 for split in ("train", "validation", "test")
              for kind in ("featuremap", "labels")}
    dataset_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    index_hash = hashlib.sha256(b"[0,1]").hexdigest()
    save_quantized(path, source, profile, {**MODEL_DEFAULTS, "readout": "flatten"}, {
        "source_sha256": "1" * 64, "dataset_sha256": dataset_hash,
        "dataset_hashes": json.dumps(hashes, sort_keys=True),
        "indices_sha256": index_hash, "calibration_indices": "[0,1]",
        "sample_count": 2, "seed": 0,
    })
    return frames, expected
def test_source_free_roundtrip_preserves_codes_and_profile(tmp_path: Path) -> None:
    # Given: a real prepared model exported without optimizer or RNG state.
    path = tmp_path / "quantized.pt"
    frames, expected = _fixture(path)
    # When: the artifact alone reconstructs the prepared model.
    loaded = load_quantized(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        before = expected.to(device)(frames.to(device))
        after = loaded.model.to(device)(frames.to(device))
    # Then: integer-backed outputs, scales, and exact stored codes agree.
    runtime = expected.blocks[0].runtime
    assert torch.equal(runtime.codes("output", before), runtime.codes("output", after))
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert (loaded.profile.to_payload() == expected.blocks[0].runtime.profile.to_payload()
            and loaded.metadata["norm_epsilon"]["blocks.0.norm"] == 1.25e-5)
    assert len(loaded.parameters) == 30
    assert sum(value.numel() for value in loaded.parameters.values()) == 15_717
    assert all(value.dtype == torch.int8 and value.device.type == "cpu"
               and value.is_contiguous() for value in loaded.parameters.values())
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"metadata", "parameters", "sha256"}
    assert not ({"optimizer_state_dict", "rng_state"} & set(raw))


@pytest.mark.parametrize("damage", ["content", "scale", "shape", "profile", "norm", "provenance", "config"])
def test_corrupt_artifact_is_rejected(tmp_path: Path, damage: str, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: one independently damaged field in an otherwise valid artifact.
    path = tmp_path / "quantized.pt"
    _fixture(path)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    name = next(iter(raw["parameters"]))
    if damage == "content":
        raw["parameters"][name].view(-1)[0] += 1
    elif damage == "scale":
        del raw["metadata"]["parameters"][name]["exponent"]
        raw["sha256"] = artifact_io._digest(raw["metadata"], raw["parameters"])
    elif damage == "shape":
        raw["parameters"][name] = raw["parameters"][name].reshape(-1)
        raw["metadata"]["parameters"][name]["sha256"] = artifact_io._tensor_hash(
            raw["parameters"][name])
        raw["sha256"] = artifact_io._digest(raw["metadata"], raw["parameters"])
    elif damage == "profile":
        key = next(key for key in raw["metadata"]["profile"] if key.endswith(".Abar"))
        raw["metadata"]["profile"][key]["exponent"] = -6
        raw["sha256"] = artifact_io._digest(raw["metadata"], raw["parameters"])
    elif damage == "norm":
        raw["metadata"]["norm_epsilon"]["blocks.0.norm"] = -1.0
        raw["sha256"] = artifact_io._digest(raw["metadata"], raw["parameters"])
    elif damage == "provenance":
        raw["metadata"]["provenance"]["calibration_indices"] = "[1,0]"
        raw["sha256"] = artifact_io._digest(raw["metadata"], raw["parameters"])
    else:
        raw["metadata"]["model_config"]["d_model"] = 1_000_000_000
        raw["sha256"] = artifact_io._digest(raw["metadata"], raw["parameters"])
    torch.save(raw, path)
    called: list[bool] = []
    if damage in ("config", "norm"):
        monkeypatch.setattr(parameter_io, "EMamba", lambda **_config: called.append(True))
    # When/Then: strict load rejects tampering before model use.
    with pytest.raises(ArtifactError):
        load_quantized(path)
    assert not called
