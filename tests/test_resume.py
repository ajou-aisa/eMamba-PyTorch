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
from training import session as training_session


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
                         (1, 128, 1e-3, 0, 0, "flatten"))
        self.assertIsNone(args.resume)

    def test_resume_uses_last_checkpoint_directory(self) -> None:
        checkpoint = Path("results/example/last.pt")
        args = self.parse("--mode", "train", "--resume", str(checkpoint),
                          "--epochs", "3")
        self.assertEqual(args.output_dir, checkpoint.resolve().parent)
        self.assertEqual(args.epochs, 3)
        self.assertIsNone(args.lr)
        self.assertIsNone(args.batch_size)

    def test_legacy_nonlinear_override_is_checkpoint_only(self) -> None:
        eval_args = self.parse("--mode", "eval", "--checkpoint", "old.pt",
                               "--legacy-nonlinear", "native_fp32")
        self.assertEqual(eval_args.legacy_nonlinear, "native_fp32")
        resume_args = self.parse("--mode", "train", "--resume", "results/example/last.pt",
                                 "--epochs", "3", "--legacy-nonlinear", "piecewise_fp32")
        self.assertEqual(resume_args.legacy_nonlinear, "piecewise_fp32")
        self.reject("--mode", "train", "--legacy-nonlinear", "native_fp32")

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
                              ("--seed", "1"), ("--readout", "flatten"),
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

    def run_training(
        self, output_dir: Path, epochs: int, resume: Path | None = None,
        fingerprint: str = "a" * 64,
    ) -> None:
        args = argparse.Namespace(
            mode="train", device="cpu", data_root=output_dir,
            output_dir=output_dir, epochs=epochs, steps=None,
            batch_size=2 if resume is None else None,
            lr=1e-3 if resume is None else None,
            seed=0 if resume is None else None,
            readout="flatten" if resume is None else None,
            num_workers=0 if resume is None else None,
            checkpoint=None, split=None, resume=resume,
            legacy_nonlinear=None,
            debug_numerics=False, no_progress=True, json_stdout=True,
        )
        with (
            patch.object(train, "parse_args", return_value=args),
            patch.object(training_session, "build_dataloaders", side_effect=self.loaders),
            patch.object(training_session, "training_data_fingerprint", return_value=fingerprint),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train.main()

    def test_split_run_matches_uninterrupted_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = root / "full"
            split = root / "split"
            self.run_training(full, 4)
            train_epoch = train.train_one_epoch

            def interrupt_after_two_epochs(*args, **kwargs):
                if args[5] == 3:
                    raise RuntimeError("training interrupted")
                return train_epoch(*args, **kwargs)

            with patch.object(train, "train_one_epoch", side_effect=interrupt_after_two_epochs):
                with self.assertRaisesRegex(RuntimeError, "training interrupted"):
                    self.run_training(split, 4)
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

    def test_resume_rejects_best_checkpoint_with_wrong_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 2)
            best_path = output_dir / "best.pt"
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            original_epoch = best["epoch"]
            best["epoch"] = 99
            torch.save(best, best_path)

            with self.assertRaisesRegex(ValueError, "best.pt"):
                self.run_training(output_dir, 3, output_dir / "last.pt")

            best["epoch"] = original_epoch
            best["training_config"]["training_data_sha256"] = "b" * 64
            torch.save(best, best_path)
            with self.assertRaisesRegex(ValueError, "best.pt"):
                self.run_training(output_dir, 3, output_dir / "last.pt")

    def test_resume_rejects_changed_training_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 1)
            with self.assertRaisesRegex(ValueError, "training data"):
                self.run_training(output_dir, 2, output_dir / "last.pt", "b" * 64)

    def test_resume_rejects_stale_scheduler_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 1)
            path = output_dir / "last.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["scheduler_state_dict"]["last_epoch"] = 0
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "scheduler state"):
                self.run_training(output_dir, 2, path)

    def test_legacy_untagged_best_allows_second_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 1)
            first_rmse = torch.load(output_dir / "best.pt", weights_only=True)[
                "best_validation_rmse_cm"
            ]
            for name in ("best.pt", "last.pt"):
                path = output_dir / name
                payload = torch.load(path, map_location="cpu", weights_only=True)
                del payload["nonlinear_policy"]
                payload["git"] = {
                    "commit": "b8c091a0b16347a94024b322682dfa0034e43565",
                    "dirty": False,
                }
                torch.save(payload, path)

            worse = {"rmse_cm": {"all": first_rmse + 1}, "mae_cm": {"all": 10.0}}
            with patch.object(train, "evaluate", return_value=worse):
                self.run_training(output_dir, 2, output_dir / "last.pt")
            self.assertNotIn("nonlinear_policy", torch.load(output_dir / "best.pt", weights_only=True))
            self.run_training(output_dir, 3, output_dir / "last.pt")

    def test_resume_restores_adamw_and_saved_learning_rate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            self.run_training(output_dir, 1)
            with patch.object(train, "train_one_epoch", wraps=train.train_one_epoch) as epoch:
                self.run_training(output_dir, 2, output_dir / "last.pt")
            optimizer = epoch.call_args.args[3]
            self.assertIs(type(optimizer), torch.optim.AdamW)
            self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.01)
            self.assertEqual(optimizer.param_groups[0]["lr"], 1e-6)
            payload = torch.load(output_dir / "last.pt", weights_only=True)
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(payload["global_step"], 6)
            self.assertEqual(payload["scheduler_epoch_offset"], 1)
            self.assertEqual(payload["scheduler_state_dict"]["last_epoch"]
                             + payload["scheduler_epoch_offset"], payload["epoch"])
            self.assertTrue(all(state["step"].item() == 6 for state in optimizer.state.values()))

    def test_resume_preserves_legacy_adam_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "run"
            with patch.object(torch.optim, "AdamW", torch.optim.Adam):
                self.run_training(output_dir, 1)
            with patch.object(train, "train_one_epoch", wraps=train.train_one_epoch) as epoch:
                self.run_training(output_dir, 2, output_dir / "last.pt")
            self.assertIs(type(epoch.call_args.args[3]), torch.optim.Adam)

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
