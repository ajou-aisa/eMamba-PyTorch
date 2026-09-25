import copy
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from models.emamba import EMamba
from models.mamba.block import EMambaBlock
from models.output_head import OutputHead
from models.patch_embedding import PatchEmbedding

from .layers import QConv1d, QLinear
from .mamba.block import QBlock
from .mamba.range_norm import QRangeNorm
from .mamba.selective_ssm import QSelectiveSSM
from .quant import QuantRuntime


def _enabled(runtime: QuantRuntime) -> bool:
    return runtime.profile is not None or runtime.observer is not None

def _linear(
    source: nn.Linear, prefix: str, input_name: str,
    output_name: str, runtime: QuantRuntime,
) -> nn.Module:
    if not _enabled(runtime):
        return source
    return QLinear(source, prefix, input_name, output_name, runtime)

def _ssm(
    source: nn.Module, prefix: str, runtime: QuantRuntime,
    continuous_a: Mapping[str, Tensor],
) -> QSelectiveSSM:
    parameter_projection = _linear(
        source.ssm_param_proj, f"{prefix}.ssm_param_proj", f"{prefix}.x",
        f"{prefix}.parameters", runtime,
    )
    delta_projection = _linear(
        source.delta_proj, f"{prefix}.delta_proj", f"{prefix}.deltaInput",
        f"{prefix}.deltaProjection", runtime,
    )
    a_name = f"{prefix}.A"
    a_value = continuous_a.get(a_name, -torch.exp(source.a_log.detach()))
    return QSelectiveSSM(parameter_projection, delta_projection, a_value, source.d_skip,
                         source.dt_rank, prefix, runtime)

def _block(
    source: EMambaBlock, index: int, runtime: QuantRuntime,
    continuous_a: Mapping[str, Tensor],
) -> QBlock:
    prefix = f"blocks.{index}"
    norm = QRangeNorm(source.norm.gamma, source.norm.beta, source.norm.eps,
                      f"{prefix}.norm", runtime)
    gate = _linear(
        source.gate_proj, f"{prefix}.gate_proj", f"{prefix}.norm.output",
        f"{prefix}.gateProjection", runtime,
    )
    input_projection = _linear(
        source.input_proj, f"{prefix}.input_proj", f"{prefix}.norm.output",
        f"{prefix}.inputProjection", runtime,
    )
    convolution = source.conv
    if _enabled(runtime):
        convolution.conv1d = QConv1d(convolution.conv1d, f"{prefix}.conv1d", runtime)
    ssm = _ssm(source.ssm, f"{prefix}.ssm", runtime, continuous_a)
    output = _linear(
        source.output_proj, f"{prefix}.output_proj", f"{prefix}.gated",
        f"{prefix}.projection", runtime,
    )
    return QBlock(norm, gate, input_projection, convolution, ssm, output,
                  source.d_model, prefix, runtime)

def prepare_components(
    source: EMamba, runtime: QuantRuntime, *,
    continuous_a: Mapping[str, Tensor] | None = None,
) -> tuple[PatchEmbedding, nn.ModuleList, OutputHead]:
    model = copy.deepcopy(source)
    overrides = {} if continuous_a is None else continuous_a

    model.patch_embedding.proj = _linear(
        model.patch_embedding.proj, "patch_embedding.proj",
        "patch_embedding.patches", "patch_embedding.output", runtime,
    )

    model.blocks = nn.ModuleList([
        _block(block, index, runtime, overrides)
        for index, block in enumerate(model.blocks)
    ])
    model.head.proj[0] = _linear(
        model.head.proj[0], "head.proj.0", "head.input", "head.hidden", runtime,
    )
    model.head.proj[2] = _linear(
        model.head.proj[2], "head.proj.2", "head.activation", "output", runtime,
    )
    return model.patch_embedding, model.blocks, model.head
