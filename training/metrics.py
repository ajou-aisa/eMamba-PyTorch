import math

import torch
from torch import Tensor


def _check_finite(value: Tensor, stage: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"MARS {stage} is nonfinite")


class MARSMetricAccumulator:
    """Accumulate MARS coordinate errors across batches on CPU in FP64."""

    def __init__(self) -> None:
        self.abs_error_sum = torch.zeros(57, dtype=torch.float64)
        self.squared_error_sum = torch.zeros(57, dtype=torch.float64)
        self.sample_count = 0

    def update(
        self, prediction: Tensor, target: Tensor, *, debug_numerics: bool = False,
    ) -> None:
        if prediction.device.type != "cpu" or target.device.type != "cpu":
            raise ValueError("MARS metric requires CPU tensors")
        if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 57:
            raise ValueError("MARS metric requires matching [B, 57] shape")
        if debug_numerics:
            _check_finite(prediction, "CPU prediction")
            _check_finite(target, "target")

        prediction_fp64 = prediction.double()
        target_fp64 = target.double()
        if debug_numerics:
            _check_finite(prediction_fp64, "FP64 prediction")
            _check_finite(target_fp64, "FP64 target")

        error = prediction_fp64 - target_fp64
        if debug_numerics:
            _check_finite(error, "evaluation error")
        batch_abs_sum = error.abs().sum(dim=0)
        batch_squared_sum = error.square().sum(dim=0)
        if debug_numerics:
            _check_finite(batch_abs_sum, "absolute-error sum")
            _check_finite(batch_squared_sum, "squared-error sum")

        self.abs_error_sum += batch_abs_sum
        self.squared_error_sum += batch_squared_sum
        if debug_numerics:
            _check_finite(self.abs_error_sum, "absolute-error accumulator")
            _check_finite(self.squared_error_sum, "squared-error accumulator")
        self.sample_count += target.shape[0]

    def compute(self) -> dict[str, dict[str, float]]:
        if self.sample_count <= 0:
            raise ValueError("MARS metric requires at least one sample")

        per_coordinate = {
            "mae_cm": self.abs_error_sum / self.sample_count,
            "rmse_cm": torch.sqrt(self.squared_error_sum / self.sample_count),
        }
        result: dict[str, dict[str, float]] = {}
        for name, coordinates in per_coordinate.items():
            _check_finite(coordinates, f"{name} coordinate metric")
            axes: dict[str, float] = {}
            for axis, start, stop in (
                ("x", 0, 19), ("y", 19, 38), ("z", 38, 57), ("all", 0, 57),
            ):
                value = float(coordinates[start:stop].mean().item() * 100.0)
                if not math.isfinite(value):
                    raise FloatingPointError(f"MARS {name} {axis} metric is nonfinite")
                axes[axis] = value
            result[name] = axes
        return result
