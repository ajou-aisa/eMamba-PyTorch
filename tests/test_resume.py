import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import train


class ResumeArgumentTests(unittest.TestCase):
    def parse(self, *options: str):
        with patch.object(sys, "argv", ["train.py", *options]):
            return train.parse_args()

    def reject(self, *options: str) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse(*options)

    def test_fresh_defaults_remain_unchanged(self) -> None:
        args = self.parse("--mode", "train")
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.seed,
                          args.num_workers, args.readout),
                         (1, 128, 1e-3, 0, 0, "mean"))
        self.assertIsNone(args.resume)

    def test_resume_uses_last_checkpoint_directory(self) -> None:
        checkpoint = Path("results/example/last.pt")
        args = self.parse("--mode", "train", "--resume", str(checkpoint),
                          "--epochs", "3")
        self.assertEqual(args.output_dir, checkpoint.resolve().parent)
        self.assertEqual(args.epochs, 3)
        self.assertIsNone(args.lr)
        self.assertIsNone(args.batch_size)

    def test_resume_rejects_invalid_mode_path_and_explicit_overrides(self) -> None:
        base = ("--mode", "train", "--resume", "results/example/last.pt",
                "--epochs", "3")
        self.reject("--mode", "smoke", "--resume", "results/example/last.pt")
        self.reject("--mode", "eval", "--resume", "results/example/last.pt",
                    "--checkpoint", "results/example/last.pt")
        self.reject("--mode", "train", "--resume", "results/example/last.pt")
        self.reject("--mode", "train", "--resume", "results/example/best.pt",
                    "--epochs", "3")
        self.reject(*base, "--output-dir", "results/other")
        for option, value in (("--lr", "0.0001"), ("--batch-size", "32"),
                              ("--seed", "1"), ("--readout", "last"),
                              ("--num-workers", "2")):
            with self.subTest(option=option):
                self.reject(*base, option, value)


class ResumeTrainingTests(unittest.TestCase):
    def loaders(self, *_args, **_kwargs):
        features = torch.arange(6 * 8 * 8 * 5, dtype=torch.float32).reshape(6, 8, 8, 5) / 1000
        targets = torch.linspace(-0.1, 0.1, 6 * 57).reshape(6, 57)
        dataset = TensorDataset(features, targets)
        return {
            "train": DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0),
            "validation": DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0),
        }

    def run_training(self, output_dir: Path, epochs: int, resume: Path | None = None) -> None:
        args = argparse.Namespace(
            mode="train", device="cpu", data_root=output_dir,
            output_dir=output_dir, epochs=epochs, steps=None,
            batch_size=2 if resume is None else None,
            lr=1e-3 if resume is None else None,
            seed=0 if resume is None else None,
            readout="mean" if resume is None else None,
            num_workers=0 if resume is None else None,
            checkpoint=None, split=None, resume=resume,
            debug_numerics=False, no_progress=True, json_stdout=True,
        )
        with (
            patch.object(train, "parse_args", return_value=args),
            patch.object(train, "build_dataloaders", side_effect=self.loaders),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train.main()

    def test_split_run_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            split = root / "split"
            self.run_training(full, 4)
            self.run_training(split, 2)
            self.run_training(split, 4, split / "last.pt")
            full_state = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
            split_state = torch.load(split / "last.pt", map_location="cpu", weights_only=True)

            for name, tensor in full_state["model_state_dict"].items():
                torch.testing.assert_close(tensor, split_state["model_state_dict"][name], rtol=0, atol=0)
            for parameter_id, state in full_state["optimizer_state_dict"]["state"].items():
                resumed = split_state["optimizer_state_dict"]["state"][parameter_id]
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, resumed[name], rtol=0, atol=0)
                    else:
                        self.assertEqual(value, resumed[name])
            self.assertEqual(full_state["global_step"], split_state["global_step"])
            self.assertEqual(full_state["best_validation_rmse_cm"],
                             split_state["best_validation_rmse_cm"])
            history = [json.loads(line) for line in (split / "history.jsonl").read_text().splitlines()]
            self.assertEqual([record["epoch"] for record in history], [1, 2, 3, 4])
            self.assertEqual([record["global_step"] for record in history], [3, 6, 9, 12])

    def test_target_epoch_must_exceed_checkpoint_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 2)
            for target in (1, 2):
                with self.subTest(target=target):
                    with self.assertRaisesRegex(ValueError, "already at epoch 2"):
                        self.run_training(output_dir, target, output_dir / "last.pt")

    def test_legacy_checkpoint_warns_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 1)
            checkpoint = output_dir / "last.pt"
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload.pop("rng_state")
            torch.save(payload, checkpoint)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                self.run_training(output_dir, 2, checkpoint)
            self.assertTrue(any("lacks RNG state" in str(item.message) for item in recorded))
            history = [json.loads(line) for line in (output_dir / "history.jsonl").read_text().splitlines()]
            self.assertEqual([record["epoch"] for record in history], [1, 2])

    def test_resume_keeps_best_until_validation_improves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 1)
            checkpoint = output_dir / "last.pt"
            first = json.loads((output_dir / "history.jsonl").read_text().splitlines()[0])
            best_rmse = first["best_validation_rmse_cm"]
            original_best = (output_dir / "best.pt").read_bytes()
            worse = {"rmse_cm": {"all": best_rmse + 1}, "mae_cm": {"all": 10.0}}
            with (
                patch.object(train, "train_one_epoch", return_value=(0.1, 6)),
                patch.object(train, "evaluate", return_value=worse),
            ):
                self.run_training(output_dir, 2, checkpoint)
            self.assertEqual((output_dir / "best.pt").read_bytes(), original_best)
            second = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertEqual(second["epoch"], 2)
            self.assertEqual(second["best_validation_rmse_cm"], best_rmse)

            better = {"rmse_cm": {"all": best_rmse - 1}, "mae_cm": {"all": 8.0}}
            with (
                patch.object(train, "train_one_epoch", return_value=(0.1, 9)),
                patch.object(train, "evaluate", return_value=better),
            ):
                self.run_training(output_dir, 3, checkpoint)
            selected = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
            self.assertEqual(selected["epoch"], 3)
            self.assertEqual(selected["best_validation_rmse_cm"], best_rmse - 1)


if __name__ == "__main__":
    unittest.main()
