import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import train


class TrainingEntryTests(unittest.TestCase):
    def test_debug_numerics_flag(self) -> None:
        with patch.object(sys, "argv", ["train.py", "--mode", "smoke", "--debug-numerics"]):
            self.assertTrue(train.parse_args().debug_numerics)

    def test_output_flags(self) -> None:
        with patch.object(sys, "argv", ["train.py", "--mode", "train",
                                        "--no-progress", "--json-stdout"]):
            args = train.parse_args()
        self.assertTrue(args.no_progress)
        self.assertTrue(args.json_stdout)

    def test_run_config_and_history_keep_raw_data(self) -> None:
        validation = {
            "split": "validation", "samples": 1, "checkpoint": "current",
            "baseline_id": "provisional_fp32_v1",
            "mae_cm": {axis: 8.26 for axis in ("x", "y", "z", "all")},
            "rmse_cm": {axis: 10.9 for axis in ("x", "y", "z", "all")},
        }
        loader = DataLoader(TensorDataset(torch.zeros(2, 8, 8, 5), torch.zeros(2, 57)))
        for json_stdout in (False, True):
            with self.subTest(json_stdout=json_stdout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = argparse.Namespace(
                    mode="train", device="cpu", seed=0, readout="mean", lr=1e-3,
                    batch_size=128, num_workers=0, data_root=root,
                    output_dir=root / "run", epochs=1, steps=None,
                    debug_numerics=False, no_progress=True, json_stdout=json_stdout,
                )
                output = io.StringIO()
                with (
                    patch.object(train, "parse_args", return_value=args),
                    patch.object(train, "build_dataloaders",
                                 return_value={"train": loader, "validation": loader}),
                    patch.object(train, "train_one_epoch", return_value=(0.308325, 1)),
                    patch.object(train, "evaluate", return_value=validation),
                    patch.object(train, "save_checkpoint"),
                    contextlib.redirect_stdout(output),
                ):
                    train.main()

                def reject(value: str) -> None:
                    raise ValueError(f"nonfinite JSON: {value}")

                config = json.loads((root / "run/run_config.json").read_text(),
                                    parse_constant=reject)
                history = json.loads((root / "run/history.jsonl").read_text(),
                                     parse_constant=reject)
                self.assertEqual(config["baseline_id"], "provisional_fp32_v1")
                self.assertEqual(config["parameter_count"], 9077)
                self.assertEqual(config["training_config"]["optimizer"], "Adam")
                self.assertEqual(history["validation"], validation)
                self.assertEqual(history["best_validation_rmse_cm"], 10.9)
                if json_stdout:
                    self.assertEqual(len(output.getvalue().splitlines()), 2)
                    self.assertEqual(json.loads(output.getvalue().splitlines()[0]), config)
                else:
                    self.assertIn("Train Loss", output.getvalue())
                    self.assertIn("Training complete", output.getvalue())

    def test_final_mae_comes_from_best_rmse_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loader = DataLoader(TensorDataset(torch.zeros(1, 8, 8, 5), torch.zeros(1, 57)))
            args = argparse.Namespace(
                mode="train", device="cpu", seed=0, readout="mean", lr=1e-3,
                batch_size=128, num_workers=0, data_root=root,
                output_dir=root / "run", epochs=2, steps=None,
                debug_numerics=False, no_progress=True, json_stdout=False,
            )
            validations = [
                {"mae_cm": {"all": 7.0}, "rmse_cm": {"all": 12.0}},
                {"mae_cm": {"all": 8.0}, "rmse_cm": {"all": 10.0}},
            ]
            output = io.StringIO()
            with (
                patch.object(train, "parse_args", return_value=args),
                patch.object(train, "build_dataloaders",
                             return_value={"train": loader, "validation": loader}),
                patch.object(train, "train_one_epoch", return_value=(0.1, 1)),
                patch.object(train, "evaluate", side_effect=validations),
                patch.object(train, "save_checkpoint"),
                contextlib.redirect_stdout(output),
            ):
                train.main()
            self.assertIn("Best epoch      2", output.getvalue())
            self.assertIn("Best MAE        8.000 cm", output.getvalue())
            self.assertNotIn("Best MAE        7.000 cm", output.getvalue())

    def test_real_split_count_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "featuremap_train.npy", np.zeros((1, 8, 8, 5)))
            np.save(root / "labels_train.npy", np.zeros((1, 57)))
            with self.assertRaisesRegex(ValueError, "train.*24066"):
                train.build_dataloaders(root, ("train",), 128, 0)

    def test_train_stops_before_checkpoint_and_history_on_nonfinite_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                mode="train", device="cpu", seed=0, readout="mean", lr=1e-3,
                batch_size=2, num_workers=0, data_root=root,
                output_dir=root / "run", epochs=2, steps=None,
                debug_numerics=False, no_progress=True, json_stdout=False,
            )
            invalid = {"rmse_cm": {"all": float("nan")}}
            with (
                patch.object(train, "parse_args", return_value=args),
                patch.object(train, "build_dataloaders",
                             return_value={"train": [], "validation": []}),
                patch.object(train, "train_one_epoch", return_value=(1.0, 1)) as train_epoch,
                patch.object(train, "evaluate", return_value=invalid),
                patch.object(train, "save_checkpoint") as save,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(FloatingPointError, "validation.*RMSE"):
                    train.main()
            self.assertEqual(train_epoch.call_count, 1)
            save.assert_not_called()
            self.assertFalse((root / "run/history.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
