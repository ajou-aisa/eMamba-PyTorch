import torch
import pytest

from models.emamba import EMamba
from models.mamba.block import EMambaBlock
from models.mamba.selective_ssm import SelectiveSSM
from models.piecewise import piecewise_exp, piecewise_silu


@pytest.mark.parametrize("use_pwl", [False, True])
def test_fp32_block_gate_uses_selected_silu(use_pwl: bool) -> None:
    block = EMambaBlock(d_model=2, expand=1, d_state=1, use_pwl=use_pwl)
    with torch.no_grad():
        block.gate_proj.weight.copy_(torch.eye(2) * 0.5)
        block.input_proj.weight.copy_(torch.eye(2))
        block.output_proj.weight.copy_(torch.eye(2))
        block.conv.conv1d.weight.zero_()
        block.conv.conv1d.weight[:, :, -1] = 1
        assert block.conv.conv1d.bias is not None
        block.conv.conv1d.bias.zero_()
        block.ssm.ssm_param_proj.weight.zero_()
    tokens = torch.tensor([[[0.25, -0.25]]])
    normalized = block.norm(tokens)
    activation = piecewise_silu if use_pwl else torch.nn.functional.silu
    expected = tokens + normalized * activation(normalized * 0.5)
    torch.testing.assert_close(block(tokens), expected, rtol=0, atol=0)


@pytest.mark.parametrize("use_pwl", [False, True])
def test_fp32_ssm_uses_selected_exp_for_abar(use_pwl: bool) -> None:
    ssm = SelectiveSSM(d_inner=1, d_state=1, dt_rank=1, use_pwl=use_pwl)
    with torch.no_grad():
        ssm.ssm_param_proj.weight.fill_(1)
        ssm.delta_proj.weight.zero_()
        ssm.delta_proj.bias.fill_(0.25)
        ssm.a_log.zero_()
        ssm.d_skip.zero_()
    tokens = torch.tensor([[[1.0], [2.0]]])
    decay = (piecewise_exp if use_pwl else torch.exp)(torch.tensor(-0.25))
    expected = torch.stack((torch.tensor(0.25), 2 * (decay * 0.25 + 1))).reshape(1, 2, 1)
    torch.testing.assert_close(ssm(tokens), expected, rtol=0, atol=0)


def test_fp32_piecewise_model_has_finite_gradients() -> None:
    torch.manual_seed(19)
    model = EMamba()
    frames = torch.randn(2, 8, 8, 5, requires_grad=True)
    result = model(frames)
    result.square().mean().backward()
    assert result.shape == (2, 57) and result.dtype == torch.float32
    assert frames.grad is not None and torch.isfinite(frames.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
