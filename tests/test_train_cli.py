import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import train
from training import session as training_session
from training import workflow as training_workflow


class TrainCliTests(unittest.TestCase):
    def test_smoke_json_writes_two_records_without_checkpoints(self) -> None:
        # Given: one in-memory batch and a fresh output directory.
        loader = DataLoader(TensorDataset(torch.zeros(1, 8, 8, 5), torch.zeros(1, 57)))
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "smoke"
            output = io.StringIO()
            with (
                patch.object(sys, "argv", [
                    "train.py", "--mode", "smoke", "--device", "cpu", "--steps", "1",
                    "--output-dir", str(output_dir), "--json-stdout",
                ]),
                patch.object(training_session, "build_dataloaders", return_value={"train": loader}),
                patch.object(training_session, "training_data_fingerprint", return_value="a" * 64),
                patch.object(training_workflow, "diagnose_delta", return_value=[]),
                patch.object(training_workflow, "train_one_epoch", return_value=(0.0, 1)),
                contextlib.redirect_stdout(output),
            ):
                # When: the smoke CLI branch runs.
                train.main()

            # Then: stdout mirrors the two files and no checkpoint is produced.
            records = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0], json.loads((output_dir / "run_config.json").read_text()))
            self.assertEqual(records[1], json.loads((output_dir / "smoke.json").read_text()))
            self.assertEqual(records[0]["mode"], "smoke")
            self.assertEqual(records[1]["steps"], 1)
            self.assertFalse(list(output_dir.glob("*.pt")))

    def test_eval_json_uses_requested_split_and_checkpoint(self) -> None:
        # Given: a requested test split and checkpoint with local I/O patched away.
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            output = io.StringIO()
            loader = DataLoader(TensorDataset(torch.zeros(1, 8, 8, 5), torch.zeros(1, 57)))
            with (
                patch.object(sys, "argv", [
                    "train.py", "--mode", "eval", "--device", "cpu", "--split", "test",
                    "--checkpoint", str(checkpoint), "--json-stdout",
                ]),
                patch.object(training_workflow, "load_checkpoint", return_value=(None, {})) as load,
                patch.object(training_workflow, "build_dataloaders", return_value={"test": loader}) as build,
                patch.object(training_workflow, "evaluate", return_value={
                    "split": "test", "checkpoint": str(checkpoint), "samples": 1,
                }) as evaluate,
                contextlib.redirect_stdout(output),
            ):
                # When: the eval CLI branch runs.
                train.main()

            # Then: the requested inputs reach evaluation and its JSON record.
            load.assert_called_once_with(checkpoint, torch.device("cpu"), legacy_nonlinear=None)
            self.assertEqual(build.call_args.args[1], ("test",))
            self.assertIs(evaluate.call_args.args[1], loader)
            self.assertEqual(evaluate.call_args.args[3:5], ("test", str(checkpoint)))
            self.assertEqual(json.loads(output.getvalue()), {
                "mode": "eval", "precision": {"tf32": "not_applicable"},
                "split": "test", "checkpoint": str(checkpoint), "samples": 1,
            })

    def test_auto_device_prefers_cuda_then_mps_then_cpu(self) -> None:
        # Given: each possible availability combination.
        for cuda, mps, expected in (
            (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"),
        ):
            with self.subTest(cuda=cuda, mps=mps):
                with (
                    patch.object(train.torch.cuda, "is_available", return_value=cuda),
                    patch.object(train.torch.backends.mps, "is_available", return_value=mps),
                ):
                    # When: the training CLI resolves its automatic device.
                    selected = train.select_device("auto")
                # Then: the highest-priority available device wins.
                self.assertEqual(selected.type, expected)


if __name__ == "__main__":
    unittest.main()
