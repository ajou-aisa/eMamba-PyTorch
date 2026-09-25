from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .quant import QuantEntry, QuantProfile, QuantRuntime

__all__ = ["QuantEntry", "QuantProfile", "QuantRuntime"]


def __getattr__(name: str) -> type:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .quant import QuantEntry, QuantProfile, QuantRuntime

    return {"QuantEntry": QuantEntry, "QuantProfile": QuantProfile,
            "QuantRuntime": QuantRuntime}[name]
