import unittest

import torch

from training.metrics import MARSMetricAccumulator


class MARSMetricAccumulatorTests(unittest.TestCase):
    def test_coordinate_mean_and_axis_order(self):
        error = torch.zeros(2, 57)
        error[:, 0] = torch.tensor([0.0, 2.0])
        error[:, 19] = 3.0
        error[:, 38] = 4.0
        metrics = MARSMetricAccumulator()

        metrics.update(error, torch.zeros_like(error))
        result = metrics.compute()

        self.assertAlmostEqual(result["mae_cm"]["x"], 100 / 19)
        self.assertAlmostEqual(result["rmse_cm"]["x"], 100 * 2 ** 0.5 / 19)
        self.assertAlmostEqual(result["rmse_cm"]["y"], 300 / 19)
        self.assertAlmostEqual(result["rmse_cm"]["z"], 400 / 19)
        self.assertAlmostEqual(result["rmse_cm"]["all"], 100 * (2 ** 0.5 + 3 + 4) / 57)
        self.assertNotAlmostEqual(result["rmse_cm"]["all"],
                                  100 * error.square().mean().sqrt().item())

    def test_unequal_batches_match_single_batch(self):
        prediction = torch.arange(3 * 57, dtype=torch.float64).reshape(3, 57) / 100
        target = torch.zeros_like(prediction)
        whole = MARSMetricAccumulator()
        split = MARSMetricAccumulator()

        whole.update(prediction, target)
        split.update(prediction[:2], target[:2])
        split.update(prediction[2:], target[2:])

        self.assertEqual(split.sample_count, 3)
        torch.testing.assert_close(split.abs_error_sum, whole.abs_error_sum)
        torch.testing.assert_close(split.squared_error_sum, whole.squared_error_sum)
        for metric in ("mae_cm", "rmse_cm"):
            for axis in ("x", "y", "z", "all"):
                self.assertAlmostEqual(split.compute()[metric][axis],
                                       whole.compute()[metric][axis])

    def test_empty_accumulator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            MARSMetricAccumulator().compute()

    def test_debug_rejects_nonfinite_cpu_input(self):
        prediction = torch.zeros(1, 57)
        target = torch.zeros_like(prediction)
        prediction[0, 0] = float("nan")
        metrics = MARSMetricAccumulator()

        with self.assertRaisesRegex(FloatingPointError, "CPU prediction is nonfinite"):
            metrics.update(prediction, target, debug_numerics=True)
        self.assertEqual(metrics.sample_count, 0)

    def test_debug_rejects_nonfinite_target(self):
        prediction = torch.zeros(1, 57)
        target = torch.full_like(prediction, float("inf"))

        with self.assertRaisesRegex(FloatingPointError, "target is nonfinite"):
            MARSMetricAccumulator().update(prediction, target, debug_numerics=True)

    def test_final_metric_rejects_nonfinite_without_debug(self):
        prediction = torch.full((1, 57), float("nan"))
        metrics = MARSMetricAccumulator()
        metrics.update(prediction, torch.zeros_like(prediction))

        with self.assertRaisesRegex(FloatingPointError, "coordinate metric is nonfinite"):
            metrics.compute()

    def test_shape_and_cpu_float64_state(self):
        metrics = MARSMetricAccumulator()
        with self.assertRaisesRegex(ValueError, "shape"):
            metrics.update(torch.zeros(2, 56), torch.zeros(2, 56))

        metrics.update(torch.ones(1, 57), torch.zeros(1, 57))
        self.assertEqual(metrics.abs_error_sum.dtype, torch.float64)
        self.assertEqual(metrics.abs_error_sum.device.type, "cpu")
        self.assertEqual(metrics.sample_count, 1)


if __name__ == "__main__":
    unittest.main()
