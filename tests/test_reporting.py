import contextlib
import io
import unittest
from pathlib import Path

from training.reporting import (
    print_epoch_header,
    print_epoch_result,
    print_eval_summary,
    print_run_summary,
    print_resume_summary,
    print_smoke_summary,
    print_training_summary,
)


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "mode": "train",
            "baseline_id": "provisional_fp32_v1",
            "device": "mps",
            "model_config": {
                "d_model": 20, "expand": 2, "num_blocks": 2,
                "d_state": 8, "readout": "mean",
            },
            "delta_config": {
                "delta_activation": "post_projection_relu",
                "dt_init_min": 0.001, "dt_init_max": 0.1,
            },
            "parameter_count": 9077,
            "fp32_parameter_bytes": 36308,
            "training_config": {
                "optimizer": "Adam", "lr": 0.001, "batch_size": 128,
                "epochs": 20, "seed": 0, "steps": None,
            },
        }
        self.validation = {
            "split": "validation", "samples": 8033,
            "checkpoint": "current", "baseline_id": "provisional_fp32_v1",
            "mae_cm": {"x": 9.0, "y": 7.0, "z": 8.78, "all": 8.26},
            "rmse_cm": {"x": 12.98, "y": 8.507, "z": 11.213, "all": 10.9},
        }

    def capture(self, function, *args, **kwargs) -> str:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            function(*args, **kwargs)
        return stream.getvalue()

    def test_run_summary_uses_config(self) -> None:
        output = self.capture(
            print_run_summary, self.config, Path("results/run"),
            {"train": 24066, "validation": 8033},
        )
        for value in (
            "provisional_fp32_v1", "mps", "9,077", "35.46 KiB",
            "Adam", "lr=0.001", "Linear → ReLU", "24,066", "8,033",
        ):
            self.assertIn(value, output)

    def test_epoch_result_precision_and_best_marker(self) -> None:
        header = self.capture(print_epoch_header)
        self.assertIn("Train Loss", header)
        self.assertIn("Val RMSE", header)
        best = self.capture(
            print_epoch_result, 16, 20, 0.013143,
            self.validation, 10.9, True, show_axis_metrics=True,
        )
        for value in ("0.013143", "8.260", "10.900", "*", "X=12.980"):
            self.assertIn(value, best)
        ordinary = self.capture(
            print_epoch_result, 17, 20, 0.012898,
            self.validation, 10.9, False,
        )
        self.assertNotIn("*", ordinary)

    def test_final_summary_pairs_best_mae_with_best_rmse(self) -> None:
        last = {**self.validation,
                "mae_cm": {**self.validation["mae_cm"], "all": 7.5},
                "rmse_cm": {**self.validation["rmse_cm"], "all": 11.093}}
        output = self.capture(
            print_training_summary, 16, self.validation, last, Path("results/run/best.pt"),
        )
        for value in ("Best epoch", "16", "8.260 cm", "10.900 cm", "11.093 cm", "best.pt"):
            self.assertIn(value, output)
        self.assertNotIn("7.500 cm", output)

    def test_smoke_and_eval_summaries(self) -> None:
        smoke_config = {**self.config, "mode": "smoke"}
        smoke = {
            "steps": 25, "initial_loss": 1.486482, "final_loss": 1.060345,
            "finite": True,
            "delta_initial": [{"mean": 0.01645, "zero_fraction": 0.0}],
            "delta_final": [{"mean": 0.03144, "zero_fraction": 0.2}],
        }
        smoke_output = self.capture(print_smoke_summary, smoke_config, smoke)
        for value in ("Smoke Test", "1.486482", "1.060345", "Delta Block 0", "20.0%", "PASS"):
            self.assertIn(value, smoke_output)

        evaluation = {**self.validation, "checkpoint": "results/run/best.pt"}
        eval_output = self.capture(print_eval_summary, evaluation, "mps")
        for value in ("Evaluation", "mps", "validation", "8,033", "MAE", "RMSE", "All"):
            self.assertIn(value, eval_output)
        for axis in ("X", "Y", "Z"):
            self.assertIn(axis, eval_output)

    def test_resume_summary_shows_continuation_and_device_change(self) -> None:
        payload = {
            "epoch": 20, "global_step": 3780,
            "best_validation_rmse_cm": 10.8999,
            "environment": {"device": "mps"},
        }
        output = self.capture(
            print_resume_summary, Path("results/run/last.pt"), payload, 150, "cuda",
        )
        for value in ("Resume", "Completed epoch  20", "Next epoch       21",
                      "Target epoch     150", "3,780", "10.900 cm", "cuda (from mps)"):
            self.assertIn(value, output)


if __name__ == "__main__":
    unittest.main()
