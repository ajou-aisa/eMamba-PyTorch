import hashlib
import json
from collections.abc import Mapping


class ProvenanceError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def validate_provenance(provenance: Mapping[str, str | int]) -> None:
    hashes = ("source_sha256", "dataset_sha256", "indices_sha256")
    required = {*hashes, "dataset_hashes", "calibration_indices", "sample_count", "seed"}
    try:
        datasets = json.loads(str(provenance["dataset_hashes"]))
        indices = json.loads(str(provenance["calibration_indices"]))
    except (json.JSONDecodeError, KeyError, TypeError):
        raise ProvenanceError("artifact provenance is invalid") from None
    expected = {f"{split}.{kind}" for split in ("train", "validation", "test")
                for kind in ("featuremap", "labels")}
    digest_values = tuple(provenance.get(key) for key in hashes)
    if isinstance(datasets, dict):
        digest_values += tuple(datasets.values())
    invalid_hash = any(type(value) is not str or len(value) != 64
                       or bool(set(value) - set("0123456789abcdef"))
                       for value in digest_values)
    dataset_identity = hashlib.sha256(
        json.dumps(datasets, sort_keys=True).encode()).hexdigest()
    indices_identity = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()).hexdigest()
    if (set(provenance) != required or invalid_hash or not isinstance(datasets, dict)
            or set(datasets) != expected or not isinstance(indices, list)
            or len(indices) != provenance.get("sample_count")
            or any(type(index) is not int or index < 0 for index in indices)
            or provenance.get("dataset_sha256") != dataset_identity
            or provenance.get("indices_sha256") != indices_identity
            or type(provenance["sample_count"]) is not int
            or type(provenance["seed"]) is not int):
        raise ProvenanceError("artifact provenance is invalid")
