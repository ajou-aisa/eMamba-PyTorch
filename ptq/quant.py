from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, TypedDict
import torch
from torch import Tensor
from .brevitas import fixed_identity
from .ops import quantize_codes
from .statistics import BoundaryStatistics, QuantStatistics
class QuantizationError(ValueError):
    pass
class QuantEntryPayload(TypedDict):
    exponent: int
    bits: int
    zero_point: int
@dataclass(frozen=True, slots=True)
class QuantEntry:
    exponent: int
    bits: int = 8
    zero_point: int = 0
    def __post_init__(self) -> None:
        valid = (type(self.exponent) is int and -126 <= self.exponent <= 127
                 and type(self.bits) is int and 2 <= self.bits <= 24)
        valid = valid and -149 <= self.exponent + self.bits - 1 <= 127
        if not valid or self.zero_point != 0:
            raise QuantizationError("invalid symmetric power-of-two quantization entry")
@dataclass(frozen=True, slots=True)
class QuantProfile:
    entries: Mapping[str, QuantEntry]
    @classmethod
    def from_entries(cls, entries: Mapping[str, QuantEntry]) -> "QuantProfile":
        if not entries:
            raise QuantizationError("quantization profile must not be empty")
        copied = dict(entries)
        for name, entry in copied.items():
            if name.endswith(".Abar") and (entry.exponent != -7 or entry.bits != 8):
                raise QuantizationError("Abar must use INT8 exponent -7")
            if name.endswith(".currentState"):
                state = copied.get(name.removesuffix("currentState") + "state")
                if (entry.bits != 24 or state is None or state.bits != 17
                        or entry.exponent + 7 != state.exponent):
                    raise QuantizationError("SSM current/stored state profile is invalid")
        return cls(MappingProxyType(copied))
    @classmethod
    def from_payload(cls, payload: Mapping[str, QuantEntryPayload]) -> "QuantProfile":
        return cls.from_entries({name: QuantEntry(**entry) for name, entry in payload.items()})
    def to_payload(self) -> dict[str, QuantEntryPayload]:
        return {name: {"exponent": entry.exponent, "bits": entry.bits,
                       "zero_point": entry.zero_point} for name, entry in self.entries.items()}
class Observer(Protocol):
    def __call__(self, name: str, value: Tensor, bits: int) -> None: ...
@dataclass(frozen=True, slots=True)
class QuantRuntime:
    profile: QuantProfile | None
    observer: Observer | None
    _stats: QuantStatistics | None = field(default=None, repr=False, compare=False)
    def __post_init__(self) -> None:
        if self.profile is not None and self.observer is not None:
            raise QuantizationError("runtime cannot profile and quantize simultaneously")
    @classmethod
    def bypass(cls) -> "QuantRuntime":
        return cls(None, None, None)
    @classmethod
    def profiling(cls, observer: Observer) -> "QuantRuntime":
        return cls(None, observer, None)
    @classmethod
    def frozen(cls, profile: QuantProfile) -> "QuantRuntime":
        return cls(profile, None, QuantStatistics())
    @property
    def is_frozen(self) -> bool:
        return self.profile is not None
    def entry(self, name: str) -> QuantEntry:
        if self.profile is None:
            raise QuantizationError("quantization profile is not frozen")
        try:
            return self.profile.entries[name]
        except KeyError as error:
            raise QuantizationError(f"missing quantization entry: {name}") from error
    def codes(self, name: str, value: Tensor) -> Tensor:
        entry = self.entry(name)
        return quantize_codes(value, entry.exponent, entry.bits)
    def record_codes(self, name: str, codes: Tensor, bits: int) -> None:
        if self._stats is not None:
            self._stats.record(name, codes, bits)
    def statistics(self) -> Mapping[str, BoundaryStatistics]:
        return {} if self._stats is None else self._stats.snapshot()
    def boundary(self, name: str, value: Tensor, bits: int = 8) -> Tensor:
        if self.observer is not None:
            self.observer(name, value, bits)
            return value
        if self.profile is None:
            return value
        entry = self.entry(name)
        if entry.bits != bits:
            raise QuantizationError(f"boundary bit width mismatch: {name}")

        # Compute and record integer codes for quantization validation
        raw = torch.round(value.detach().to(torch.float64) / (2.0 ** entry.exponent)).to(torch.int64)
        self.record_codes(name, raw, bits)

        # Quantize each element of the input tensor
        return fixed_identity(entry.exponent, entry.bits, value.device)(value)
