import io
import unittest

import torch

from models.emamba import EMamba
from models.mamba.block import EMambaBlock
from models.mamba.range_norm import RangeNorm
from models.output_head import OutputHead
from models.patch_embedding import PatchEmbedding


class PatchEmbeddingTests(unittest.TestCase):
    def test_patch_values_follow_row_major_order(self) -> None:
        frames = torch.arange(32, dtype=torch.float32).reshape(1, 4, 4, 2)
        patch = PatchEmbedding(in_channels=2, patch_size=2, d_model=8)

        tokens = patch(frames)

        expected = torch.stack(
            [frames[0, row:row + 2, col:col + 2].flatten()
             for row in (0, 2) for col in (0, 2)]
        ).unsqueeze(0)
        torch.testing.assert_close(tokens, expected)

    def test_rejects_invalid_configuration_and_frames(self) -> None:
        for settings in ((0, 2, 0), (2, 0, 0), (2, 2, 0), (2, 2, 7)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                PatchEmbedding(*settings)

        patch = PatchEmbedding(2, 2, 8)
        for frames in (torch.zeros(4, 4, 2), torch.zeros(1, 3, 4, 2),
                       torch.zeros(1, 4, 4, 1)):
            with self.subTest(shape=tuple(frames.shape)), self.assertRaises(ValueError):
                patch(frames)


class RangeNormTests(unittest.TestCase):
    def test_constant_input_is_finite(self) -> None:
        norm = RangeNorm(3)
        output = norm(torch.full((2, 4, 3), 5.0))
        torch.testing.assert_close(output, torch.zeros_like(output))
        self.assertTrue(torch.isfinite(output).all())

    def test_matches_range_formula_and_has_finite_backward(self) -> None:
        norm = RangeNorm(3)
        tokens = torch.tensor([[[1.0, 3.0, 6.0]]], requires_grad=True)

        output = norm(tokens)
        output.square().sum().backward()

        expected = torch.tensor([[[-7 / 15, -1 / 15, 8 / 15]]])
        torch.testing.assert_close(output, expected)
        assert tokens.grad is not None
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_rejects_invalid_settings_and_width(self) -> None:
        for d_model, eps in ((0, 1e-6), (3, 0.0), (3, -1.0)):
            with self.subTest(d_model=d_model, eps=eps), self.assertRaises(ValueError):
                RangeNorm(d_model, eps)

        with self.assertRaises(ValueError):
            RangeNorm(3)(torch.zeros(2, 4, 2))
        self.assertIn("eps=", repr(RangeNorm(3)))


class OutputHeadTests(unittest.TestCase):
    def test_mean_and_last_readout(self) -> None:
        tokens = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
        mean = OutputHead(2, 1, "mean")
        last = OutputHead(2, 1, "last")
        with torch.no_grad():
            for head in (mean, last):
                head.proj.weight.fill_(1.0)
                head.proj.bias.zero_()

        torch.testing.assert_close(mean(tokens), torch.tensor([[8.0]]))
        torch.testing.assert_close(last(tokens), torch.tensor([[12.0]]))

    def test_rejects_invalid_input_and_has_finite_backward(self) -> None:
        head = OutputHead(2, 3)
        for tokens in (torch.zeros(2, 2), torch.zeros(2, 0, 2),
                       torch.zeros(2, 4, 3)):
            with self.subTest(shape=tuple(tokens.shape)), self.assertRaises(ValueError):
                head(tokens)
        with self.assertRaises(ValueError):
            OutputHead(2, 3, "invalid")

        tokens = torch.randn(2, 4, 2, requires_grad=True)
        head(tokens).square().mean().backward()
        assert tokens.grad is not None
        self.assertTrue(torch.isfinite(tokens.grad).all())


class EMambaTests(unittest.TestCase):
    def test_rejects_empty_block_stack(self) -> None:
        with self.assertRaises(ValueError):
            EMamba(num_blocks=0)

    def test_block_rejects_invalid_settings_and_tokens(self) -> None:
        for settings in ((0, 2, 8), (20, 0, 8), (20, 2, 0)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                EMambaBlock(*settings)

        block = EMambaBlock(20, 2, 8)
        for tokens in (torch.zeros(2, 20), torch.zeros(2, 0, 20),
                       torch.zeros(2, 4, 19)):
            with self.subTest(shape=tuple(tokens.shape)), self.assertRaises(ValueError):
                block(tokens)

    def test_default_and_last_readout_shapes(self) -> None:
        frames = torch.randn(2, 8, 8, 5)
        for readout in ("mean", "last"):
            with self.subTest(readout=readout):
                model = EMamba(readout=readout)
                self.assertEqual(model.head.readout, readout)
                self.assertEqual(tuple(model(frames).shape), (2, 57))

    def test_full_model_backward_and_state_dict_roundtrip(self) -> None:
        torch.manual_seed(0)
        model = EMamba()
        frames = torch.randn(2, 8, 8, 5)
        output = model(frames)
        output.square().mean().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                            if p.grad is not None))

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = EMamba()
        restored.load_state_dict(torch.load(buffer, map_location="cpu", weights_only=True))
        torch.testing.assert_close(restored(frames), output)


if __name__ == "__main__":
    unittest.main()
