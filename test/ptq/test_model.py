import copy

import torch
from torch import nn

from models.emamba import EMamba
from models.q_emamba import QEMamba
from models.mamba.block import MambaConv1D
from models.mamba.q_block import QBlock
from ptq.calibrate import ProfileObserver, candidate_profiles
from ptq.model import prepare_model
from ptq.ops import quantize_codes
from ptq.quant import QuantRuntime


def test_bypass_returns_quantized_model_entry_and_preserves_output() -> None:
    torch.manual_seed(3)
    source = EMamba()
    frames = torch.randn(2, 8, 8, 5)
    expected = source(frames)
    prepared = prepare_model(source, QuantRuntime.bypass())
    actual = prepared(frames)
    assert type(prepared) is QEMamba
    assert type(prepared.patch_embedding) is type(source.patch_embedding)
    assert type(prepared.head) is type(source.head)
    assert isinstance(prepared.patch_embedding.proj, nn.Linear)
    assert isinstance(prepared.blocks[0], QBlock)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_prepare_model_does_not_mutate_source_and_keeps_causal_conv() -> None:
    source = EMamba()
    state = copy.deepcopy(source.state_dict())
    prepared = prepare_model(source, QuantRuntime.bypass())
    assert isinstance(prepared.blocks[0].conv, MambaConv1D)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)


def test_frozen_model_runs_with_profiled_brevitas_boundaries() -> None:
    source, frames = EMamba(), torch.randn(1, 8, 8, 5)
    observer = ProfileObserver()
    prepare_model(source, QuantRuntime.profiling(observer))(frames)
    profile = candidate_profiles(observer.snapshot())["max"]
    runtime = QuantRuntime.frozen(profile)
    prepared = prepare_model(source, runtime)
    result = prepared(frames)
    assert result.shape == (1, 57)
    assert torch.isfinite(result).all()
    assert runtime.codes("output", result).dtype == torch.int64
    weight_name = "patch_embedding.proj.weight"
    brevitas_codes = prepared.patch_embedding.proj.layer.quant_weight().int().to(torch.int64)
    expected_codes = quantize_codes(
        source.patch_embedding.proj.weight, profile.entries[weight_name].exponent,
    )
    torch.testing.assert_close(brevitas_codes, expected_codes, rtol=0, atol=0)
