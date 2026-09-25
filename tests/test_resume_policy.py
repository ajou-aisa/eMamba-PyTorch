import json
import tempfile
import unittest
from pathlib import Path

from training.resume import validate_resume_files, validate_training_config


class ResumePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        training = {
            "optimizer": "Adam", "loss": "MSELoss", "precision": "fp32",
            "lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 0,
            "gradient_clip": 1.0, "batch_size": 128, "seed": 0,
            "num_workers": 0, "epochs": 3, "steps": None,
            "tf32": {"tf32": "not_applicable"},
        }
        self.payload = {
            "baseline_id": "provisional_fp32_v1",
            "model_config": {"readout": "mean"},
            "delta_config": {"delta_activation": "post_projection_relu"},
            "parameter_count": 9077, "fp32_parameter_bytes": 36308,
            "training_config": training,
            "epoch": 3, "global_step": 9, "best_validation_rmse_cm": 10.5,
        }
        self.records = [
            self.record(1, 3, 12.0, 12.0),
            self.record(2, 6, 10.5, 10.5),
            self.record(3, 9, 11.0, 10.5),
        ]
        self.write_history()

    @staticmethod
    def record(epoch: int, step: int, rmse: float, best: float) -> dict:
        return {
            "epoch": epoch, "global_step": step,
            "train_loss": 0.1,
            "validation": {
                "mae_cm": {"all": rmse - 2},
                "rmse_cm": {"all": rmse},
            },
            "best_validation_rmse_cm": best,
        }

    def write_history(self) -> None:
        (self.root / "history.jsonl").write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in self.records)
        )

    def write_run_config(self) -> None:
        config = {
            key: self.payload[key] for key in (
                "baseline_id", "model_config", "delta_config",
                "parameter_count", "fp32_parameter_bytes", "training_config",
            )
        }
        config.update({"mode": "train", "device": "cpu"})
        (self.root / "run_config.json").write_text(json.dumps(config, allow_nan=False))

    def test_matching_history_finds_best_checkpoint_metric_pair(self) -> None:
        best_epoch, best_validation, config_present = validate_resume_files(
            self.root, self.payload,
        )

        self.assertEqual(best_epoch, 2)
        self.assertEqual(best_validation, self.records[1]["validation"])
        self.assertFalse(config_present)

    def test_existing_run_config_accepts_tuple_list_betas_and_device_change(self) -> None:
        self.write_run_config()
        config_path = self.root / "run_config.json"
        config = json.loads(config_path.read_text())
        config["device"] = "mps"
        config["training_config"]["epochs"] = 150
        config_path.write_text(json.dumps(config))

        _, _, config_present = validate_resume_files(self.root, self.payload)

        self.assertTrue(config_present)

    def test_history_epoch_step_and_best_metric_must_match_checkpoint(self) -> None:
        for field, invalid in (
            ("epoch", 4), ("global_step", 10),
            ("best_validation_rmse_cm", 10.4),
        ):
            with self.subTest(field=field):
                payload = dict(self.payload)
                payload[field] = invalid
                with self.assertRaisesRegex(ValueError, "history.*checkpoint"):
                    validate_resume_files(self.root, payload)

    def test_missing_or_invalid_history_is_rejected(self) -> None:
        history_path = self.root / "history.jsonl"
        history_path.unlink()
        with self.assertRaisesRegex(ValueError, "history.jsonl"):
            validate_resume_files(self.root, self.payload)
        history_path.write_text("{broken}\n")
        with self.assertRaisesRegex(ValueError, "history.jsonl"):
            validate_resume_files(self.root, self.payload)

    def test_nonfinite_history_and_run_config_are_rejected(self) -> None:
        history_path = self.root / "history.jsonl"
        history_path.write_text(history_path.read_text().replace("11.0", "NaN"))
        with self.assertRaisesRegex(ValueError, "nonfinite|NaN"):
            validate_resume_files(self.root, self.payload)
        self.write_history()
        self.write_run_config()
        config_path = self.root / "run_config.json"
        config_path.write_text(config_path.read_text().replace("0.001", "Infinity"))
        with self.assertRaisesRegex(ValueError, "nonfinite|Infinity"):
            validate_resume_files(self.root, self.payload)

    def test_run_config_mismatch_is_rejected(self) -> None:
        self.write_run_config()
        config_path = self.root / "run_config.json"
        original = json.loads(config_path.read_text())
        for field, invalid in (
            ("baseline_id", "wrong"), ("model_config", {"readout": "last"}),
            ("delta_config", {"delta_activation": "pre_projection_relu"}),
            ("parameter_count", 1), ("fp32_parameter_bytes", 1),
        ):
            with self.subTest(field=field):
                config = {**original, field: invalid}
                config_path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, f"run_config.*{field}"):
                    validate_resume_files(self.root, self.payload)
        config = original.copy()
        config["training_config"] = {**original["training_config"], "lr": 0.01}
        config_path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "run_config.*lr"):
            validate_resume_files(self.root, self.payload)

    def test_training_config_requires_supported_settings(self) -> None:
        self.assertEqual(validate_training_config(self.payload), self.payload["training_config"])
        for field, invalid in (
            ("optimizer", "SGD"), ("loss", "L1Loss"),
            ("precision", "fp16"), ("gradient_clip", 0.0),
        ):
            with self.subTest(field=field):
                payload = dict(self.payload)
                payload["training_config"] = {**self.payload["training_config"], field: invalid}
                with self.assertRaisesRegex(ValueError, field):
                    validate_training_config(payload)
        payload = dict(self.payload)
        payload["training_config"] = {"optimizer": "Adam"}
        with self.assertRaisesRegex(ValueError, "training_config"):
            validate_training_config(payload)

        payload = dict(self.payload)
        payload["training_config"] = {**self.payload["training_config"],
                                      "training_data_sha256": None}
        with self.assertRaisesRegex(ValueError, "training_data_sha256"):
            validate_training_config(payload)


if __name__ == "__main__":
    unittest.main()
