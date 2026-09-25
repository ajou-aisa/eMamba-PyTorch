import math
from collections.abc import Mapping

import torch
from torch import Tensor

from models.emamba import EMamba
from models.q_emamba import QEMamba

from .ops import dequantize_codes, quantize_codes
from .quant import QuantProfile, QuantRuntime
class ParameterArtifactError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def export_parameters(
    source: EMamba, profile: QuantProfile,
) -> tuple[dict[str, Tensor], dict[str, dict]]:
    parameters, descriptions = {}, {}
    for source_name, value in source.state_dict().items():
        name, scale_name, derived, exported = source_name, source_name, None, value.detach()
        if source_name.endswith(".a_log"):
            name, derived = source_name.removesuffix("a_log") + "A", source_name
            scale_name = name
            exported = -torch.exp(exported)
        elif source_name.endswith(".d_skip"):
            name, derived = source_name.removesuffix("d_skip") + "D", source_name
            scale_name = name
        elif ".conv.conv1d." in source_name:
            scale_name = source_name.replace(".conv.conv1d.", ".conv1d.")
        entry = profile.entries[scale_name]
        codes = quantize_codes(exported, entry.exponent).to(torch.int8).cpu().contiguous()
        parameters[name] = codes
        descriptions[name] = {
            "source_name": source_name, "scale_name": scale_name,
            "exponent": entry.exponent,
            "shape": list(codes.shape), "derived_from": derived,
        }
    return parameters, descriptions


def restore_parameters(
    model_config: Mapping[str, int | str], parameters: Mapping[str, Tensor],
    descriptions: Mapping[str, dict], profile: QuantProfile,
    norm_epsilon: Mapping[str, float], *, use_pwl: bool = False,
) -> QEMamba:
    num_blocks = model_config.get("num_blocks")
    expected_norms = ({f"blocks.{index}.norm" for index in range(num_blocks)}
                      if type(num_blocks) is int else set())
    if set(norm_epsilon) != expected_norms or not all(
        type(value) in (int, float) and math.isfinite(value) and value > 0.0
        for value in norm_epsilon.values()
    ):
        raise ParameterArtifactError("artifact norm epsilon is invalid")
    source, overrides = EMamba(**model_config), {}
    state = source.state_dict()
    source_names = {item.get("source_name") for item in descriptions.values()}
    if source_names != set(state):
        raise ParameterArtifactError("artifact source-name mapping is invalid")
    for index, block in enumerate(source.blocks):
        block.norm.eps = norm_epsilon[f"blocks.{index}.norm"]
    for name, value in parameters.items():
        description = descriptions[name]
        source_name, scale_name = description["source_name"], description["scale_name"]
        exponent = description["exponent"]
        expected_derived = source_name if name.endswith((".A", ".D")) else None
        expected_name = source_name
        expected_scale = source_name.replace(".conv.conv1d.", ".conv1d.")
        if name.endswith(".A"):
            expected_name = source_name.removesuffix("a_log") + "A"
            expected_scale = expected_name
        elif name.endswith(".D"):
            expected_name = source_name.removesuffix("d_skip") + "D"
            expected_scale = expected_name
        if (name != expected_name or scale_name != expected_scale
                or description["derived_from"] != expected_derived
                or tuple(value.shape) != tuple(state[source_name].shape)
                or profile.entries[scale_name].exponent != exponent):
            raise ParameterArtifactError(f"artifact mapping for {name} is invalid")
        restored = dequantize_codes(value, exponent)
        if name.endswith(".A"):
            overrides[name] = restored
        else:
            state[source_name] = restored
    source.load_state_dict(state, strict=True)
    return QEMamba(source, QuantRuntime.frozen(profile, use_pwl=use_pwl), continuous_a=overrides)
