import hashlib
import importlib.metadata
import json
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple
import torch
from torch import Tensor
from models.emamba import EMamba
from models.q_emamba import QEMamba
from ptq.parameters import export_parameters, restore_parameters
from ptq.provenance import validate_provenance
from ptq.quant import QuantProfile
from ptq.piecewise import PIECEWISE_SPEC
from training.checkpoint import MODEL_DEFAULTS
_FORMAT = "emamba-ptq-v1"
_POLICY = {"scale": "power_of_two", "zero_point": 0, "rounding": "ties_to_even",
           "clipping": "signed_saturating", "nonlinear": "native_fp32"}
_PWL_POLICY = {**_POLICY, "nonlinear": "piecewise_fp32", "piecewise": PIECEWISE_SPEC}
_STATE = {"current_bits": 24, "stored_bits": 17, "right_shift": 7,
          "output_uses_current": True}
class ArtifactError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
class LoadedArtifact(NamedTuple):
    model: QEMamba
    profile: QuantProfile
    metadata: dict
    parameters: Mapping[str, Tensor]
def _tensor_hash(value: Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
def _digest(metadata: dict, parameters: Mapping[str, Tensor]) -> str:
    checksum = hashlib.sha256(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())
    for name, value in sorted(parameters.items()):
        checksum.update(name.encode())
        checksum.update(value.contiguous().numpy().tobytes())
    return checksum.hexdigest()
def save_quantized(
    path: Path, source: EMamba, profile: QuantProfile,
    model_config: Mapping[str, int | str], provenance: Mapping[str, str | int], use_pwl: bool = False,
) -> str:
    validate_provenance(provenance)
    parameters, descriptions = export_parameters(source, profile)
    for name, value in parameters.items():
        descriptions[name]["sha256"] = _tensor_hash(value)
    metadata = {
        "format": _FORMAT, "model_config": dict(model_config), "parameters": descriptions,
        "profile": profile.to_payload(), "norm_epsilon": {
            f"blocks.{index}.norm": block.norm.eps for index, block in enumerate(source.blocks)},
        "quantization": _PWL_POLICY if use_pwl else _POLICY,
        "state": _STATE, "provenance": dict(provenance),
        "versions": {"torch": str(torch.__version__),
                     "brevitas": importlib.metadata.version("brevitas")},
    }
    digest = _digest(metadata, parameters)
    torch.save({"metadata": metadata, "parameters": parameters, "sha256": digest}, path)
    return digest
def load_quantized(path: Path) -> LoadedArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "parameters", "sha256"}:
        raise ArtifactError("artifact envelope is invalid")
    metadata, parameters = payload["metadata"], payload["parameters"]
    if not isinstance(metadata, dict) or not isinstance(parameters, dict):
        raise ArtifactError("artifact contents are invalid")
    required = {"format", "model_config", "parameters", "profile", "norm_epsilon",
                "quantization", "state", "provenance", "versions"}
    if (set(metadata) != required or metadata["format"] != _FORMAT
            or metadata["model_config"] != {**MODEL_DEFAULTS, "readout": "flatten"}):
        raise ArtifactError("artifact metadata is invalid")
    if metadata["quantization"] not in (_POLICY, _PWL_POLICY) or metadata["state"] != _STATE:
        raise ArtifactError("artifact numeric policy is invalid")
    descriptions = metadata["parameters"]
    if not isinstance(descriptions, dict) or set(descriptions) != set(parameters):
        raise ArtifactError("artifact parameter metadata is invalid")
    for name, value in parameters.items():
        description = descriptions[name]
        if (not isinstance(value, Tensor) or value.dtype != torch.int8 or value.device.type != "cpu"
                or not value.is_contiguous() or not isinstance(description, dict)
                or list(value.shape) != description.get("shape")
                or _tensor_hash(value) != description.get("sha256")):
            raise ArtifactError(f"artifact parameter {name} is invalid")
        try:
            description["source_name"], description["scale_name"]
            description["exponent"], description["derived_from"]
        except (KeyError, TypeError):
            raise ArtifactError(f"artifact parameter {name} scale is invalid") from None
    if len(parameters) != 30 or sum(value.numel() for value in parameters.values()) != 15_717:
        raise ArtifactError("artifact parameter count is invalid")
    if payload["sha256"] != _digest(metadata, parameters):
        raise ArtifactError("artifact hash mismatch")
    try:
        validate_provenance(metadata["provenance"])
        profile = QuantProfile.from_payload(metadata["profile"])
        model = restore_parameters(
            metadata["model_config"], parameters, descriptions, profile,
            metadata["norm_epsilon"], use_pwl=metadata["quantization"] == _PWL_POLICY,
        )
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ArtifactError(f"artifact reconstruction failed: {error}") from error
    return LoadedArtifact(model, profile, metadata, parameters)
