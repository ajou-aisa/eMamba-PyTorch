import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import train


class TrainingEntryTests(unittest.TestCase):
    def test_debug_numerics_flag(self) -> None:
        with patch.object(sys, "argv", ["train.py", "--mode", "smoke", "--debug-numerics"]):
            self.assertTrue(train.parse_args().debug_numerics)

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
                output_dir=root / "run", epochs=2, steps=None, debug_numerics=False,
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
