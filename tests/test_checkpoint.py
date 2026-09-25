import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from models.emamba import EMamba, NonlinearPolicy
from training.checkpoint import (
    BASELINE_ID,
    MODEL_DEFAULTS,
    capture_rng_state,
    load_checkpoint,
    parameter_size,
    restore_optimizer_state,
    restore_rng_state,
    save_checkpoint,
)


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_config = {**MODEL_DEFAULTS, "readout": "flatten"}
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
        self.assertEqual(payload["nonlinear_policy"], "piecewise_fp32")
        self.assertEqual(restored.nonlinear_policy, "piecewise_fp32")
        self.assertEqual(payload["readout"], "flatten")
        self.assertEqual(payload["parameter_count"], parameter_size(self.model)[0])
        self.assertEqual(payload["fp32_parameter_bytes"], parameter_size(self.model)[1])
        self.assertIn("optimizer_state_dict", payload)
        self.assertIn("environment", payload)
        self.assertIn("git", payload)

    def test_native_checkpoint_roundtrip_preserves_output(self) -> None:
        model = EMamba(**self.model_config, nonlinear_policy="native_fp32")
        frames = torch.randn(2, 8, 8, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "native.pt"
            save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 2, 1.0,
                            self.model_config, {"seed": 0}, torch.device("cpu"))
            restored, payload = load_checkpoint(path, torch.device("cpu"))
        self.assertEqual(payload["nonlinear_policy"], "native_fp32")
        self.assertEqual(restored.nonlinear_policy, "native_fp32")
        torch.testing.assert_close(restored(frames), model(frames), rtol=0, atol=0)

    def test_untagged_clean_checkpoint_uses_historical_policy(self) -> None:
        commits: tuple[tuple[str, NonlinearPolicy], ...] = (
            ("007ed66d386d366c90322d88288ecfb7e6fcbeee", "native_fp32"),
            ("6c8e19c7c8cc06a742a17a2b768903646ce1c3e3", "native_fp32"),
            ("b8c091a0b16347a94024b322682dfa0034e43565", "piecewise_fp32"),
        )
        frames = torch.randn(2, 8, 8, 5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            for commit, policy in commits:
                with self.subTest(policy=policy):
                    model = EMamba(**self.model_config, nonlinear_policy=policy)
                    save_checkpoint(path, model, torch.optim.Adam(model.parameters()), 1, 2, 1.0,
                                    self.model_config, {"seed": 0}, torch.device("cpu"))
                    payload = torch.load(path, map_location="cpu", weights_only=True)
                    del payload["nonlinear_policy"]
                    payload["git"] = {"commit": commit, "dirty": False}
                    torch.save(payload, path)
                    restored, _ = load_checkpoint(path, torch.device("cpu"))
                    self.assertEqual(restored.nonlinear_policy, policy)
                    torch.testing.assert_close(restored(frames), model(frames), rtol=0, atol=0)

    def test_untagged_ambiguous_checkpoint_requires_explicit_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            self.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            del payload["nonlinear_policy"]
            payload["git"] = {
                "commit": "007ed66d386d366c90322d88288ecfb7e6fcbeee", "dirty": True,
            }
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "ambiguous nonlinear policy"):
                load_checkpoint(path, torch.device("cpu"))
            restored, _ = load_checkpoint(
                path, torch.device("cpu"), legacy_nonlinear="piecewise_fp32",
            )
        self.assertEqual(restored.nonlinear_policy, "piecewise_fp32")

    def test_rejects_invalid_or_conflicting_nonlinear_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            self.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["nonlinear_policy"] = "unknown"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "nonlinear policy"):
                load_checkpoint(path, torch.device("cpu"))
            payload["nonlinear_policy"] = "piecewise_fp32"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "nonlinear policy"):
                load_checkpoint(path, torch.device("cpu"), legacy_nonlinear="native_fp32")

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
        self.assertEqual(restored.head.readout, "flatten")

    def test_rejects_inconsistent_metadata(self) -> None:
        changes = (
            ("baseline_id", "provisional_fp32_v1", "baseline_id"),
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

    def test_rng_state_roundtrip_is_weights_only_safe(self) -> None:
        original_python = random.getstate()
        original_numpy = np.random.get_state()
        original_torch = torch.get_rng_state()
        try:
            random.seed(29)
            np.random.seed(29)
            torch.manual_seed(29)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "last.pt"
                save_checkpoint(
                    path, self.model, self.optimizer, 1, 2, 1.0,
                    self.model_config, {"seed": 29}, torch.device("cpu"),
                    rng_state=capture_rng_state(),
                )
                expected = (random.random(), np.random.random(), torch.rand(4))
                payload = torch.load(path, map_location="cpu", weights_only=True)
                restore_rng_state(payload["rng_state"], torch.device("cpu"))
                actual = (random.random(), np.random.random(), torch.rand(4))
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])
            torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
            self.assertIn("device_rng_support", payload["rng_state"])
        finally:
            random.setstate(original_python)
            np.random.set_state(original_numpy)
            torch.set_rng_state(original_torch)

    def test_optimizer_state_roundtrip_on_cpu(self) -> None:
        frames = torch.randn(2, 8, 8, 5)
        self.optimizer.zero_grad(set_to_none=True)
        self.model(frames).square().mean().backward()
        self.optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            self.save(path)
            restored_model, payload = load_checkpoint(path, torch.device("cpu"))
            restored_optimizer = torch.optim.Adam(restored_model.parameters())
            restore_optimizer_state(restored_optimizer, payload, torch.device("cpu"))

        for original, restored in zip(self.model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(original, restored, rtol=0, atol=0)
            original_state = self.optimizer.state[original]
            restored_state = restored_optimizer.state[restored]
            for key in ("step", "exp_avg", "exp_avg_sq"):
                self.assertEqual(restored_state[key].device.type, "cpu")
                torch.testing.assert_close(original_state[key], restored_state[key], rtol=0, atol=0)

    def test_optimizer_state_requires_valid_payload(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            restore_optimizer_state(self.optimizer, {}, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "optimizer state"):
            restore_optimizer_state(
                self.optimizer, {"optimizer_state_dict": {"state": {}}}, torch.device("cpu"),
            )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_rng_state_restores_mps_next_draw(self) -> None:
        original = capture_rng_state()
        try:
            torch.mps.manual_seed(29)
            state = capture_rng_state()
            expected = torch.rand(4, device="mps").cpu()
            restore_rng_state(state, torch.device("mps"))
            actual = torch.rand(4, device="mps").cpu()
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        finally:
            restore_rng_state(original, torch.device("mps"))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS unavailable")
    def test_optimizer_state_moves_to_mps(self) -> None:
        frames = torch.randn(2, 8, 8, 5)
        self.optimizer.zero_grad(set_to_none=True)
        self.model(frames).square().mean().backward()
        self.optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            self.save(path)
            restored_model, payload = load_checkpoint(path, torch.device("mps"))
            restored_optimizer = torch.optim.Adam(restored_model.parameters())
            restore_optimizer_state(restored_optimizer, payload, torch.device("mps"))
        for state in restored_optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device.type, "mps")


if __name__ == "__main__":
    unittest.main()
