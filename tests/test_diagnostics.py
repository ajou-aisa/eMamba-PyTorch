import math
import unittest
from unittest.mock import patch

import torch

from models.emamba import EMamba
from models.mamba.block import EMambaBlock
from training.diagnostics import diagnose_delta


def hook_counts(model: EMamba) -> list[int]:
    counts = []
    for block in model.blocks:
        assert isinstance(block, EMambaBlock)
        counts.append(len(block.ssm._forward_pre_hooks))
    return counts


class DeltaDiagnosticsTests(unittest.TestCase):
    def test_reports_finite_values_without_leaking_hooks(self) -> None:
        torch.manual_seed(0)
        model = EMamba()
        frames = torch.randn(2, 8, 8, 5, requires_grad=True)
        before = hook_counts(model)

        stats = diagnose_delta(model, frames)

        self.assertEqual(len(stats), len(model.blocks))
        expected = {
            "min", "max", "mean", "zero_fraction",
            "all_zero_channel_fraction", "a_bar_max",
        }
        for block_stats in stats:
            self.assertEqual(set(block_stats), expected)
            self.assertTrue(all(isinstance(value, float) and math.isfinite(value)
                                for value in block_stats.values()))
            self.assertGreaterEqual(block_stats["min"], 0)
            self.assertLessEqual(block_stats["a_bar_max"], 1)
        self.assertEqual(hook_counts(model), before)

    def test_removes_hooks_when_forward_fails(self) -> None:
        model = EMamba()
        before = hook_counts(model)

        with patch.object(model, "forward", side_effect=RuntimeError("forced failure")):
            with self.assertRaisesRegex(RuntimeError, "forced failure"):
                diagnose_delta(model, torch.randn(1, 8, 8, 5))

        self.assertEqual(hook_counts(model), before)


if __name__ == "__main__":
    unittest.main()
