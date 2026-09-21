from functools import partial

import torch
from torch import Tensor, nn

from models.emamba import EMamba
from models.mamba.block import EMambaBlock
from models.mamba.selective_ssm import SelectiveSSM


def diagnose_delta(model: EMamba, frames: Tensor) -> list[dict[str, float]]:
    stats: list[dict[str, float]] = [{} for _ in model.blocks]
    hooks = []

    def inspect(module: nn.Module, inputs: tuple[Tensor], index: int) -> None:
        assert isinstance(module, SelectiveSSM)
        with torch.no_grad():
            delta = module.compute_delta(inputs[0])
            continuous_a = -torch.exp(module.a_log)
            stats[index] = {
                "min": delta.min().item(), "max": delta.max().item(),
                "mean": delta.mean().item(),
                "zero_fraction": (delta == 0).float().mean().item(),
                "all_zero_channel_fraction": (delta == 0).all(dim=(0, 1)).float().mean().item(),
                "a_bar_max": torch.exp(delta.unsqueeze(-1) * continuous_a).max().item(),
            }

    for index, block in enumerate(model.blocks):
        assert isinstance(block, EMambaBlock)
        hooks.append(block.ssm.register_forward_pre_hook(partial(inspect, index=index)))
    try:
        with torch.no_grad():
            model(frames)
    finally:
        for hook in hooks:
            hook.remove()
    return stats
