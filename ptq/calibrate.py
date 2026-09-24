"""Deterministic train-only PTQ calibration."""
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Final, NamedTuple
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset

from .quant import QuantEntry, QuantProfile
SAMPLE_LIMIT: Final = 65_536
_PARAMETER_SUFFIXES: Final = (".weight", ".bias", ".gamma", ".beta", ".A", ".D")
class CalibrationError(ValueError):
    def __init__(self, name: str, reason: str) -> None:
        super().__init__(f"{name}: {reason}")

class Observation(NamedTuple):
    count: int
    maximum: float
    sample: Tensor
    bits: int

class ProfileObserver:
    """Mutable accumulator required while a profiling runtime is active."""
    def __init__(self, sample_limit: int = SAMPLE_LIMIT) -> None:
        if sample_limit < 2:
            raise CalibrationError("sample_limit", "must be at least two")
        self._limit = sample_limit
        self._values: dict[str, Observation] = {}
    def __call__(self, name: str, value: Tensor, bits: int) -> None:
        flat = value.detach().to(device="cpu", dtype=torch.float64).abs().reshape(-1)
        if flat.numel() == 0 or not torch.isfinite(flat).all():
            raise CalibrationError(name, "observation is empty or nonfinite")
        old = self._values.get(name)
        if old is not None and old.bits != bits:
            raise CalibrationError(name, "bit width changed during calibration")
        sample = flat if old is None else torch.cat((old.sample, flat))
        if sample.numel() > self._limit:
            positions = torch.linspace(0, sample.numel() - 1, self._limit,
                                       dtype=torch.float64).floor().long()
            sample = sample[positions]
        count = flat.numel() + (0 if old is None else old.count)
        maximum = float(flat.max()) if old is None else max(old.maximum, float(flat.max()))
        self._values[name] = Observation(count, maximum, sample, bits)
    def snapshot(self) -> Mapping[str, Observation]:
        return {name: Observation(item.count, item.maximum, item.sample.clone(), item.bits)
                for name, item in self._values.items()}

def select_indices(dataset_size: int, count: int = 2_048,
                   seed: int = 0) -> tuple[int, ...]:
    if dataset_size <= 0 or count <= 0 or count > dataset_size:
        raise CalibrationError("indices", "count must fit a nonempty dataset")
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randperm(dataset_size, generator=generator)[:count].tolist())
def index_hash(indices: Sequence[int]) -> str:
    encoded = json.dumps(list(indices), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
def _exponent(threshold: float, denominator: int, name: str, offset: int = 0) -> int:
    if not math.isfinite(threshold) or threshold < 0.0:
        raise CalibrationError(name, "threshold or exponent is unsupported")
    exponent = (0 if threshold == 0.0 else math.ceil(math.log2(threshold / denominator))) + offset
    if -126 <= exponent <= 127:
        return exponent
    raise CalibrationError(name, "threshold or exponent is unsupported")
def candidate_profiles(observations: Mapping[str, Observation]) -> Mapping[str, QuantProfile]:
    if not observations:
        raise CalibrationError("profile", "no observations were collected")
    profiles: dict[str, QuantProfile] = {}
    for mode in ("max", "percentile"):
        entries: dict[str, QuantEntry] = {}
        for name, item in sorted(observations.items()):
            if name in entries:
                continue
            if name.endswith(".currentState"):
                current = _exponent(item.maximum, 65_535, name, offset=-7)
                stored = current + 7
                entries[name] = QuantEntry(current, bits=24)
                entries[name.removesuffix("currentState") + "state"] = QuantEntry(stored, bits=17)
                continue
            maximum_only = name.endswith(_PARAMETER_SUFFIXES) or name.endswith(".Abar")
            threshold = item.maximum if mode == "max" or maximum_only else float(
                torch.quantile(item.sample, 0.999, interpolation="linear"))
            exponent = -7 if name.endswith(".Abar") else _exponent(threshold, 127, name)
            entries[name] = QuantEntry(exponent, bits=item.bits)
        profiles[mode] = QuantProfile.from_entries(entries)
    return profiles
def profile_hash(profile: QuantProfile) -> str:
    encoded = json.dumps(profile.to_payload(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
def collect_calibration(model: nn.Module, dataset: Dataset[tuple[Tensor, Tensor]],
                        indices: Sequence[int], device: torch.device | str = "cpu",
                        batch_size: int = 64) -> None:
    if not indices or batch_size <= 0:
        raise CalibrationError("collection", "indices and batch size must be positive")
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for features, _labels in loader:
            model(features.to(device))
