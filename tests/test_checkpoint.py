import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.emamba import EMamba
from training.checkpoint import (
    BASELINE_ID,
    MODEL_DEFAULTS,
    load_checkpoint,
    parameter_size,
    save_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = {**MODEL_DEFAULTS, "readout": "last"}
        self.model = EMamba(**self.model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save(self, path: Path) -> None:
        save_checkpoint(
            path, self.model, self.optimizer, 1, 2, 1.0,
            self.model_config, {"seed": 0}, torch.device("cpu"),
        )

    def test_roundtrip_preserves_output_and_payload(self) -> None:
        frames = torch.randn(2, 8, 8, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.save(path)
            restored, payload = load_checkpoint(path, torch.device("cpu"))

        torch.testing.assert_close(restored(frames), self.model(frames))
        self.assertEqual(payload["baseline_id"], BASELINE_ID)
        self.assertEqual(payload["readout"], "last")
        self.assertEqual(payload["parameter_count"], parameter_size(self.model)[0])
        self.assertEqual(payload["fp32_parameter_bytes"], parameter_size(self.model)[1])
        self.assertIn("optimizer_state_dict", payload)
        self.assertIn("environment", payload)
        self.assertIn("git", payload)

    def test_load_uses_cpu_map_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.save(path)
            original_load = torch.load
            original_state_load = EMamba.load_state_dict
            original_to = EMamba.to
            calls = []

            def record_state_load(model, state, *, strict=True, assign=False):
                calls.append(("load_state_dict", next(model.parameters()).device.type))
                return original_state_load(model, state, strict=strict, assign=assign)

            def record_to(model, *args, **kwargs):
                calls.append(("to", next(model.parameters()).device.type))
                return original_to(model, *args, **kwargs)

            with patch("training.checkpoint.torch.load", wraps=original_load) as load:
                with patch.object(EMamba, "load_state_dict", record_state_load):
                    with patch.object(EMamba, "to", record_to):
                        load_checkpoint(path, torch.device("cpu"))

        self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
        self.assertIs(load.call_args.kwargs["weights_only"], True)
        self.assertEqual(calls, [("load_state_dict", "cpu"), ("to", "cpu")])

    def test_loads_old_infinite_best_metric_for_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["best_validation_rmse_cm"] = float("inf")
            torch.save(payload, path)
            restored, loaded = load_checkpoint(path, torch.device("cpu"))

        self.assertEqual(loaded["best_validation_rmse_cm"], float("inf"))
        self.assertEqual(restored.head.readout, "last")

    def test_rejects_inconsistent_metadata(self) -> None:
        changes = (
            ("readout", "mean", "readout"),
            ("delta_config", {"delta_activation": "pre_projection_relu"}, "delta"),
            ("parameter_count", 0, "parameter count"),
            ("fp32_parameter_bytes", 0, "parameter count"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.save(path)
            original = torch.load(path, map_location="cpu", weights_only=True)
            for field, invalid, message in changes:
                with self.subTest(field=field):
                    payload = original.copy()
                    payload[field] = invalid
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_checkpoint(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
