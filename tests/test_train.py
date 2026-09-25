import argparse
import contextlib
import io
import json
import math
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
            "baseline_id": "provisional_fp32_v2_flatten",
            "mae_cm": {axis: 8.26 for axis in ("x", "y", "z", "all")},
            "rmse_cm": {axis: 10.9 for axis in ("x", "y", "z", "all")},
        }
        loader = DataLoader(TensorDataset(torch.zeros(2, 8, 8, 5), torch.zeros(2, 57)))
        for json_stdout in (False, True):
            with self.subTest(json_stdout=json_stdout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                args = argparse.Namespace(
                    mode="train", device="cpu", seed=0, readout="flatten", lr=1e-3,
                    batch_size=128, num_workers=0, data_root=root,
                    output_dir=root / "run", epochs=1, steps=None,
                    debug_numerics=False, no_progress=True, json_stdout=json_stdout,
                    resume=None,
                )
                output = io.StringIO()
                with (
                    patch.object(train, "parse_args", return_value=args),
                    patch.object(train, "build_dataloaders",
                                 return_value={"train": loader, "validation": loader}),
                    patch.object(train, "training_data_fingerprint", return_value="a" * 64),
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
                self.assertEqual(config["baseline_id"], "provisional_fp32_v2_flatten")
                self.assertEqual(config["parameter_count"], 15717)
                self.assertEqual(config["training_config"]["optimizer"], "AdamW")
                self.assertEqual(config["training_config"]["weight_decay"], 0.01)
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
                mode="train", device="cpu", seed=0, readout="flatten", lr=1e-3,
                batch_size=128, num_workers=0, data_root=root,
                output_dir=root / "run", epochs=2, steps=None,
                debug_numerics=False, no_progress=True, json_stdout=False,
                resume=None,
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
                patch.object(train, "training_data_fingerprint", return_value="a" * 64),
                patch.object(train, "train_one_epoch", return_value=(0.1, 1)),
                patch.object(train, "evaluate", side_effect=validations),
                patch.object(train, "save_checkpoint"),
                contextlib.redirect_stdout(output),
            ):
                train.main()
            self.assertIn("Best epoch      2", output.getvalue())
            self.assertIn("Best MAE        8.000 cm", output.getvalue())
            self.assertNotIn("Best MAE        7.000 cm", output.getvalue())

    def test_cosine_lr_and_early_stopping_keep_final_checkpoint(self) -> None:
        loader = DataLoader(TensorDataset(torch.zeros(2, 8, 8, 5), torch.zeros(2, 57)))
        for epochs, improvement_epoch, expected_last in ((2, 1, 2), (100, 1, 16), (100, 5, 20)):
            with self.subTest(epochs=epochs, improvement_epoch=improvement_epoch):
                with tempfile.TemporaryDirectory() as directory:
                    output_dir = Path(directory) / "run"
                    metrics = [
                        {"rmse_cm": {"all": 9.0 if epoch >= improvement_epoch else 10.0},
                         "mae_cm": {"all": 8.0}}
                        for epoch in range(1, epochs + 1)
                    ]
                    with (
                        patch.object(sys, "argv", [
                            "train.py", "--mode", "train", "--device", "cpu",
                            "--epochs", str(epochs), "--output-dir", str(output_dir),
                            "--json-stdout",
                        ]),
                        patch.object(train, "build_dataloaders",
                                     return_value={"train": loader, "validation": loader}),
                        patch.object(train, "training_data_fingerprint", return_value="a" * 64),
                        patch.object(train, "evaluate", side_effect=metrics),
                        contextlib.redirect_stdout(io.StringIO()),
                    ):
                        train.main()
                    last = torch.load(output_dir / "last.pt", weights_only=True)
                    best = torch.load(output_dir / "best.pt", weights_only=True)
                    history = (output_dir / "history.jsonl").read_text().splitlines()
                    expected_lr = 1e-6 + (1e-3 - 1e-6) * (
                        1 + math.cos(math.pi * expected_last / epochs)
                    ) / 2
                    self.assertEqual(last["epoch"], expected_last)
                    self.assertEqual(best["epoch"], improvement_epoch)
                    self.assertEqual(len(history), expected_last)
                    self.assertEqual(last["training_config"]["optimizer"], "AdamW")
                    group = last["optimizer_state_dict"]["param_groups"][0]
                    self.assertEqual(group["weight_decay"], 0.01)
                    self.assertAlmostEqual(group["lr"], expected_lr, places=12)

    def test_real_split_count_is_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "featuremap_train.npy", np.zeros((1, 8, 8, 5)))
            np.save(root / "labels_train.npy", np.zeros((1, 57)))
            with self.assertRaisesRegex(ValueError, "train.*24066"):
                train.build_dataloaders(root, ("train",), 128, 0)

    def test_split_ratio_coverage_pairs_and_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offset = 0
            for suffix, count in (("train", 24066), ("validate", 8033), ("test", 7984)):
                ids = np.arange(offset, offset + count, dtype=np.float32)
                features = np.zeros((count, 8, 8, 5), dtype=np.float32)
                labels = np.zeros((count, 57), dtype=np.float32)
                features[:, 0, 0, 0] = ids
                labels[:, 0] = ids
                np.save(root / f"featuremap_{suffix}.npy", features)
                np.save(root / f"labels_{suffix}.npy", labels)
                offset += count

            expected = {"train": 25679, "validation": 6420, "test": 7984}
            rng_before = torch.get_rng_state()
            loaders = train.build_dataloaders(root, tuple(expected), 128, 0)
            self.assertEqual(train.SPLIT_SIZES, expected)
            torch.testing.assert_close(torch.get_rng_state(), rng_before)
            memberships: dict[str, set[float]] = {}
            for split, loader in loaders.items():
                ids_seen: list[float] = []
                for features, labels in loader:
                    torch.testing.assert_close(features[:, 0, 0, 0], labels[:, 0])
                    ids_seen.extend(labels[:, 0].tolist())
                memberships[split] = set(ids_seen)
                self.assertEqual(len(ids_seen), expected[split])
                self.assertEqual(len(memberships[split]), expected[split])
                if split == "test":
                    self.assertEqual(ids_seen, list(range(32099, 40083)))
            self.assertFalse(memberships["train"] & memberships["validation"])
            self.assertFalse(memberships["train"] & memberships["test"])
            self.assertFalse(memberships["validation"] & memberships["test"])
            self.assertEqual(set.union(*memberships.values()), set(range(40083)))
            self.assertEqual(
                memberships["train"] | memberships["validation"], set(range(32099)),
            )

            torch.manual_seed(987)
            repeated = train.build_dataloaders(root, ("validation", "test"), 128, 0)
            for split, loader in repeated.items():
                ids_seen = [sample_id for _, labels in loader
                            for sample_id in labels[:, 0].tolist()]
                self.assertEqual(set(ids_seen), memberships[split])

    def test_train_stops_before_checkpoint_and_history_on_nonfinite_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                mode="train", device="cpu", seed=0, readout="flatten", lr=1e-3,
                batch_size=2, num_workers=0, data_root=root,
                output_dir=root / "run", epochs=2, steps=None,
                debug_numerics=False, no_progress=True, json_stdout=False,
                resume=None,
            )
            invalid = {"rmse_cm": {"all": float("nan")}}
            with (
                patch.object(train, "parse_args", return_value=args),
                patch.object(train, "build_dataloaders",
                             return_value={"train": [], "validation": []}),
                patch.object(train, "training_data_fingerprint", return_value="a" * 64),
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
