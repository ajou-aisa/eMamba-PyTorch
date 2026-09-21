import contextlib
import io
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.emamba import EMamba
from training.engine import evaluate, train_one_epoch


class TrainingEngineTests(unittest.TestCase):
    def test_progress_reports_batch_loss(self) -> None:
        model = nn.Linear(1, 1)
        loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)))
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            loss, steps = train_one_epoch(
                model, loader, nn.MSELoss(), optimizer, torch.device("cpu"), 1, 0,
                show_progress=True, progress_desc="Epoch 1/1",
            )
        self.assertEqual(steps, 2)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertIn("Epoch 1/1", output.getvalue())
        self.assertIn("loss=", output.getvalue())

    def test_criterion_and_small_last_batch_are_weighted(self) -> None:
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        loader = DataLoader(
            TensorDataset(torch.ones(3, 1), torch.tensor([[1.0], [1.0], [4.0]])),
            batch_size=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        loss, step = train_one_epoch(
            model, loader, nn.L1Loss(), optimizer, torch.device("cpu"), 1, 0,
        )
        self.assertAlmostEqual(loss, 2.0)
        self.assertEqual(step, 2)

    def test_shape_mismatch_and_nonfinite_loss(self) -> None:
        model = nn.Linear(1, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(TensorDataset(torch.zeros(1, 1), torch.zeros(1, 1)))
        with self.assertRaisesRegex(ValueError, "prediction shape"):
            train_one_epoch(model, loader, nn.MSELoss(), optimizer, torch.device("cpu"), 1, 0)

        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(TensorDataset(torch.zeros(1, 1),
                                          torch.full((1, 1), float("nan"))))
        with self.assertRaisesRegex(FloatingPointError, "loss is nonfinite"):
            train_one_epoch(model, loader, nn.MSELoss(), optimizer, torch.device("cpu"), 1, 0)

    def test_debug_prediction_and_finite_backward(self) -> None:
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        bad = DataLoader(TensorDataset(torch.tensor([[float("nan")]]), torch.zeros(1, 1)))
        with self.assertRaisesRegex(FloatingPointError, "epoch=2 batch=1 step=6.*prediction"):
            train_one_epoch(
                model, bad, nn.MSELoss(), optimizer, torch.device("cpu"), 2, 5,
                debug_numerics=True,
            )
        good = DataLoader(TensorDataset(torch.ones(2, 1), torch.zeros(2, 1)), batch_size=2)
        loss, step = train_one_epoch(
            model, good, nn.MSELoss(), optimizer, torch.device("cpu"), 2, 5,
            debug_numerics=True,
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(step, 6)
        self.assertTrue(all(torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters() if parameter.grad is not None))


class EvaluationEngineTests(unittest.TestCase):
    def test_unequal_batches_match_full_batch(self) -> None:
        torch.manual_seed(4)
        model = EMamba()
        dataset = TensorDataset(torch.randn(3, 8, 8, 5), torch.randn(3, 57))
        small = evaluate(model, DataLoader(dataset, batch_size=2),
                         torch.device("cpu"), "validation", "current")
        full = evaluate(model, DataLoader(dataset, batch_size=3),
                        torch.device("cpu"), "validation", "current")
        self.assertEqual(small["samples"], 3)
        for metric in ("mae_cm", "rmse_cm"):
            for axis in ("x", "y", "z", "all"):
                self.assertAlmostEqual(small[metric][axis], full[metric][axis])

    def test_cpu_transfer_precedes_fp64_conversion(self) -> None:
        model = EMamba()
        dataset = TensorDataset(torch.randn(2, 8, 8, 5), torch.zeros(2, 57))
        original_to = torch.Tensor.to

        def reject_combined_transfer(tensor, *args, **kwargs):
            if args and args[0] == "cpu" and kwargs.get("dtype") == torch.float64:
                raise AssertionError("combined CPU transfer and FP64 conversion")
            return original_to(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "to", reject_combined_transfer):
            result = evaluate(model, DataLoader(dataset), torch.device("cpu"),
                              "validation", "current")
        self.assertEqual(result["samples"], 2)

    def test_nonfinite_target_reports_batch_in_debug(self) -> None:
        model = EMamba()
        dataset = TensorDataset(torch.zeros(1, 8, 8, 5),
                                torch.full((1, 57), float("nan")))
        with self.assertRaisesRegex(FloatingPointError,
                                    "validation batch=1: MARS target is nonfinite"):
            evaluate(model, DataLoader(dataset), torch.device("cpu"),
                     "validation", "current", debug_numerics=True)
        with self.assertRaisesRegex(FloatingPointError, "coordinate metric is nonfinite"):
            evaluate(model, DataLoader(dataset), torch.device("cpu"),
                     "validation", "current")


if __name__ == "__main__":
    unittest.main()
