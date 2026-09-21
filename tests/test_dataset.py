import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.mars import MARSDataset


class TestMARSDataset(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.feature_path = self.root / "featuremap_train.npy"
        self.label_path = self.root / "labels_train.npy"

    def save_arrays(self, features, labels):
        np.save(self.feature_path, features)
        np.save(self.label_path, labels)

    def test_loads_unscaled_fp32_values(self):
        features = np.arange(2 * 8 * 8 * 5, dtype=np.float64).reshape(2, 8, 8, 5)
        labels = np.full((2, 57), 0.125, dtype=np.float64)
        self.save_arrays(features, labels)

        dataset = MARSDataset(self.feature_path, self.label_path)

        self.assertEqual(len(dataset), 2)
        sample, target = dataset[1]
        self.assertEqual(sample.dtype, torch.float32)
        self.assertEqual(target.dtype, torch.float32)
        torch.testing.assert_close(sample, torch.from_numpy(features[1].astype(np.float32)))
        torch.testing.assert_close(target, torch.full((57,), 0.125))

    def test_rejects_invalid_shapes_and_sample_counts(self):
        valid_features = np.zeros((2, 8, 8, 5), dtype=np.float32)
        valid_labels = np.zeros((2, 57), dtype=np.float32)
        cases = (
            (np.zeros((2, 8, 8, 4), dtype=np.float32), valid_labels),
            (valid_features, np.zeros((2, 56), dtype=np.float32)),
            (valid_features, np.zeros((1, 57), dtype=np.float32)),
            (np.zeros((0, 8, 8, 5), dtype=np.float32), np.zeros((0, 57), dtype=np.float32)),
        )
        for features, labels in cases:
            with self.subTest(feature_shape=features.shape, label_shape=labels.shape):
                self.save_arrays(features, labels)
                with self.assertRaisesRegex(ValueError, "train"):
                    MARSDataset(self.feature_path, self.label_path)

    def test_rejects_nonfinite_values_with_split_and_path(self):
        for array_name, bad_value in (("feature", np.nan), ("label", np.inf)):
            with self.subTest(array_name=array_name):
                features = np.zeros((1, 8, 8, 5), dtype=np.float32)
                labels = np.zeros((1, 57), dtype=np.float32)
                if array_name == "feature":
                    features[0, 0, 0, 0] = bad_value
                    bad_path = self.feature_path
                else:
                    labels[0, 0] = bad_value
                    bad_path = self.label_path
                self.save_arrays(features, labels)
                with self.assertRaisesRegex(ValueError, f"train.*{bad_path}"):
                    MARSDataset(self.feature_path, self.label_path)

    def test_missing_file_error_identifies_split_and_path(self):
        with self.assertRaisesRegex(ValueError, f"train.*{self.feature_path}"):
            MARSDataset(self.feature_path, self.label_path)
