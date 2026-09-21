import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train
from models.emamba import EMamba


class MetricTests(unittest.TestCase):
    def test_coordinate_mean_and_axis_order(self):
        errors = np.zeros((2, 57), dtype=np.float64)
        errors[:, 0] = [0.0, 2.0]
        errors[:, 19] = [3.0, 3.0]
        errors[:, 38] = [4.0, 4.0]
        result = train.compute_mars_metrics(
            np.abs(errors).sum(axis=0), np.square(errors).sum(axis=0), 2
        )
        self.assertAlmostEqual(result["mae_cm"]["x"], 100 / 19)
        self.assertAlmostEqual(result["rmse_cm"]["x"], 100 * np.sqrt(2) / 19)
        self.assertAlmostEqual(result["rmse_cm"]["y"], 300 / 19)
        self.assertAlmostEqual(result["rmse_cm"]["z"], 400 / 19)
        self.assertAlmostEqual(
            result["rmse_cm"]["all"], 100 * (np.sqrt(2) + 3 + 4) / 57
        )
        self.assertNotAlmostEqual(
            result["rmse_cm"]["all"],
            100 * np.sqrt(np.square(errors).mean()),
        )

    def test_unequal_batches_and_empty_input(self):
        errors = np.arange(3 * 57, dtype=np.float64).reshape(3, 57) / 100
        direct = train.compute_mars_metrics(
            np.abs(errors).sum(axis=0), np.square(errors).sum(axis=0), 3
        )
        split = train.compute_mars_metrics(
            np.abs(errors[:2]).sum(axis=0) + np.abs(errors[2:]).sum(axis=0),
            np.square(errors[:2]).sum(axis=0) + np.square(errors[2:]).sum(axis=0),
            3,
        )
        self.assertEqual(direct, split)
        with self.assertRaises(ValueError):
            train.compute_mars_metrics(np.zeros(57), np.zeros(57), 0)


class PipelineTests(unittest.TestCase):
    def test_real_split_count_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "featuremap_train.npy", np.zeros((1, 8, 8, 5)))
            np.save(root / "labels_train.npy", np.zeros((1, 57)))
            with self.assertRaisesRegex(ValueError, "train.*24066"):
                train.build_dataloaders(root, ("train",), 128, 0)

    def test_epoch_weights_last_small_batch(self):
        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        loader = DataLoader(
            TensorDataset(torch.ones(3, 1), torch.tensor([[1.0], [1.0], [4.0]])),
            batch_size=2,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        loss, step = train.train_one_epoch(model, loader, optimizer, torch.device("cpu"), 1, 0)
        self.assertAlmostEqual(loss, 6.0)
        self.assertEqual(step, 2)

    def test_evaluate_accumulates_unequal_batches(self):
        torch.manual_seed(4)
        model = EMamba()
        features = torch.randn(3, 8, 8, 5)
        targets = torch.randn(3, 57)
        dataset = TensorDataset(features, targets)
        small_batches = train.evaluate(
            model, DataLoader(dataset, batch_size=2), torch.device("cpu"),
            "validation", "current",
        )
        full_batch = train.evaluate(
            model, DataLoader(dataset, batch_size=3), torch.device("cpu"),
            "validation", "current",
        )
        self.assertEqual(small_batches, full_batch)
        self.assertEqual(small_batches["samples"], 3)

    def test_nonfinite_prediction_reports_step(self):
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = DataLoader(TensorDataset(torch.tensor([[float("nan")]]),
                                                torch.zeros(1, 1)))
        with self.assertRaisesRegex(FloatingPointError, "epoch=2 batch=1 step=6.*prediction"):
            train.train_one_epoch(model, loader, optimizer, torch.device("cpu"), 2, 5)

    def test_checkpoint_reload_and_metadata_guard(self):
        model_config = {
            "d_model": 20, "expand": 2, "patch_size": 2, "num_blocks": 2,
            "d_state": 8, "out_dim": 57, "in_channels": 5, "readout": "last",
        }
        model = EMamba(**model_config)
        optimizer = torch.optim.Adam(model.parameters())
        frames = torch.randn(2, 8, 8, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            train.save_checkpoint(
                path, model, optimizer, 1, 2, 1.0, model_config,
                {"seed": 0}, torch.device("cpu"),
            )
            restored, payload = train.load_checkpoint(path, torch.device("cpu"))
            torch.testing.assert_close(restored(frames), model(frames))
            self.assertEqual(payload["model_config"]["readout"], "last")
            payload["readout"] = "mean"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "readout"):
                train.load_checkpoint(path, torch.device("cpu"))
            payload["readout"] = "last"
            payload["delta_config"]["delta_activation"] = "pre_projection_relu"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "delta"):
                train.load_checkpoint(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
