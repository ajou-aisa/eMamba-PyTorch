from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class BoundaryStatistics:
    values: int
    clipped: int
    minimum_code: int
    maximum_code: int
    required_signed_bits: int


def _required_bits(minimum: int, maximum: int) -> int:
    positive = max(1, maximum.bit_length() + 1)
    negative = max(1, max(0, -minimum - 1).bit_length() + 1)
    return max(positive, negative)


class QuantStatistics:
    def __init__(self) -> None:
        self._entries: dict[str, BoundaryStatistics] = {}

    def record(self, name: str, codes: Tensor, bits: int) -> None:
        if codes.numel() == 0:
            return
        minimum, maximum = int(codes.min()), int(codes.max())
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        clipped = int(((codes < lower) | (codes > upper)).sum())
        previous = self._entries.get(name)
        if previous is not None:
            minimum, maximum = min(previous.minimum_code, minimum), max(previous.maximum_code, maximum)
            values, clipped = previous.values + codes.numel(), previous.clipped + clipped
        else:
            values = codes.numel()
        self._entries[name] = BoundaryStatistics(
            values, clipped, minimum, maximum, _required_bits(minimum, maximum),
        )

    def snapshot(self) -> Mapping[str, BoundaryStatistics]:
        return dict(self._entries)
