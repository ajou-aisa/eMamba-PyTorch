import unittest

import torch

from models.mamba.block import MambaConv1D


class MambaConvTests(unittest.TestCase):
    def test_causality_and_output_length(self) -> None:
        conv = MambaConv1D(d_inner=1, d_conv=4)
        with torch.no_grad():
            conv.conv1d.weight.fill_(1.0)
            bias = conv.conv1d.bias
            assert bias is not None
            bias.zero_()
        original = torch.arange(1, 7, dtype=torch.float32).reshape(1, 6, 1)
        changed = original.clone()
        changed[0, 4, 0] = 100

        first = conv(original)
        second = conv(changed)

        self.assertEqual(tuple(first.shape), (1, 6, 1))
        torch.testing.assert_close(first[:, :4], second[:, :4])

    def test_depthwise_channels_are_independent(self) -> None:
        conv = MambaConv1D(d_inner=2, d_conv=3)
        with torch.no_grad():
            conv.conv1d.weight.fill_(1.0)
            bias = conv.conv1d.bias
            assert bias is not None
            bias.zero_()
        tokens = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])

        output = conv(tokens)

        torch.testing.assert_close(output[0, :, 0], torch.tensor([1.0, 3.0, 6.0]))
        torch.testing.assert_close(output[0, :, 1], torch.zeros(3))

    def test_backward_is_finite(self) -> None:
        tokens = torch.randn(2, 5, 3, requires_grad=True)
        conv = MambaConv1D(d_inner=3)
        conv(tokens).square().sum().backward()
        assert tokens.grad is not None
        self.assertTrue(torch.isfinite(tokens.grad).all())

    def test_rejects_invalid_settings_and_tokens(self) -> None:
        for d_inner, d_conv in ((0, 4), (2, 0)):
            with self.subTest(d_inner=d_inner, d_conv=d_conv), self.assertRaises(ValueError):
                MambaConv1D(d_inner, d_conv)

        conv = MambaConv1D(2)
        for tokens in (torch.zeros(2, 4), torch.zeros(2, 0, 2),
                       torch.zeros(2, 4, 3)):
            with self.subTest(shape=tuple(tokens.shape)), self.assertRaises(ValueError):
                conv(tokens)


if __name__ == "__main__":
    unittest.main()
