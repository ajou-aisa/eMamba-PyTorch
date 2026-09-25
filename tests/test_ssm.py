import math
import unittest

import torch

from models.mamba.selective_ssm import SelectiveSSM
from models.piecewise import piecewise_exp


class SelectiveSSMTests(unittest.TestCase):
    def test_rejects_invalid_dimensions_and_tokens(self) -> None:
        for dimensions in ((0, 2, 1), (2, 0, 1), (2, 2, 0)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                SelectiveSSM(*dimensions)

        model = SelectiveSSM(2, 2, 1)
        for tokens in (torch.zeros(2, 2), torch.zeros(2, 1, 3), torch.zeros(2, 0, 2)):
            with self.subTest(shape=tuple(tokens.shape)), self.assertRaises(ValueError):
                model(tokens)

    def test_delta_initialization_stays_in_configured_ranges(self) -> None:
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                model = SelectiveSSM(8, 3, 2)
                bound = model.dt_init_scale / math.sqrt(model.dt_rank)
                weight = model.delta_proj.weight.detach()
                bias = model.delta_proj.bias.detach()
                self.assertTrue(torch.all(weight.abs() <= bound))
                self.assertTrue(torch.any(weight != 0))
                self.assertTrue(torch.all(bias >= model.dt_init_min))
                self.assertTrue(torch.all(bias <= model.dt_init_max))

    def test_final_delta_uses_post_projection_relu(self) -> None:
        model = SelectiveSSM(2, 2, 1)
        with torch.no_grad():
            model.ssm_param_proj.weight.zero_()
            model.ssm_param_proj.weight[0, 0] = -1
            model.delta_proj.weight.fill_(-1)
            model.delta_proj.bias.zero_()
        delta = model.compute_delta(torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]))
        torch.testing.assert_close(delta, torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))

    def test_recurrence_matches_independent_reference(self) -> None:
        model = SelectiveSSM(2, 2, 1)
        with torch.no_grad():
            model.ssm_param_proj.weight[0].fill_(-0.5)
            model.ssm_param_proj.weight[1:].fill_(0.5)
            model.delta_proj.weight.fill_(-1)
            model.delta_proj.bias.fill_(-0.25)
        tokens = torch.tensor(
            [[[1.0, 1.0], [1.0, 2.0], [2.0, 1.0]],
             [[2.0, 2.0], [1.0, 3.0], [3.0, 1.0]]]
        )
        parameters = model.ssm_param_proj(tokens)
        features, input_b, output_c = parameters.split((1, 2, 2), dim=-1)
        delta = torch.relu(model.delta_proj(features))
        a = -model.a_log.exp()
        state = torch.zeros(2, 2, 2)
        expected = []
        for step in range(3):
            transition = piecewise_exp(delta[:, step, :, None] * a[None])
            injection = delta[:, step, :, None] * input_b[:, step, None, :] * tokens[:, step, :, None]
            state = transition * state + injection
            expected.append((state * output_c[:, step, None, :]).sum(-1) + model.d_skip * tokens[:, step])
        torch.testing.assert_close(model(tokens), torch.stack(expected, dim=1))

        a_bar = piecewise_exp(delta[..., None] * a)
        self.assertTrue(torch.all((a_bar >= 0) & (a_bar <= 1)))

    def test_forward_backward_are_finite_and_state_resets(self) -> None:
        for seed in range(5):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                model = SelectiveSSM(4, 3, 2)
                tokens = torch.randn(2, 4, 4, requires_grad=True)
                first = model(tokens)
                self.assertEqual(first.shape, tokens.shape)
                self.assertTrue(torch.isfinite(first).all())
                first.square().mean().backward()
                assert tokens.grad is not None
                self.assertTrue(torch.isfinite(tokens.grad).all())
                for parameter in model.parameters():
                    assert parameter.grad is not None
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                model(torch.randn(2, 4, 4))
                torch.testing.assert_close(model(tokens), first)


if __name__ == "__main__":
    unittest.main()
