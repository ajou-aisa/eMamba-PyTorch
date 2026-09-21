from collections.abc import Iterable

import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from .checkpoint import BASELINE_ID
from .metrics import MARSMetricAccumulator


GRADIENT_CLIP = 1.0


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    *,
    debug_numerics: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[float, int]:
    model.train()
    loss_sum = 0.0
    elements = 0
    batches = tqdm(
        loader, desc=progress_desc or f"Epoch {epoch}",
        leave=False, dynamic_ncols=True, disable=not show_progress,
    )
    for batch_index, (features, target) in enumerate(batches, start=1):
        features = features.to(device)
        target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        location = f"epoch={epoch} batch={batch_index} step={global_step + 1}"
        if prediction.shape != target.shape:
            raise ValueError(f"{location} prediction shape {prediction.shape} != {target.shape}")
        if debug_numerics and not torch.isfinite(prediction).all():
            raise FloatingPointError(f"{location} prediction is nonfinite")
        loss = criterion(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{location} loss is nonfinite")
        loss.backward()
        if debug_numerics and any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise FloatingPointError(f"{location} gradient is nonfinite")
        try:
            clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP, error_if_nonfinite=True)
        except RuntimeError as error:
            raise FloatingPointError(f"{location} gradient norm is nonfinite") from error
        optimizer.step()
        batch_loss = loss.item()
        loss_sum += batch_loss * target.numel()
        elements += target.numel()
        global_step += 1
        if show_progress:
            batches.set_postfix(loss=f"{batch_loss:.5f}")
    if elements == 0:
        raise ValueError(f"epoch={epoch} has no training samples")
    return loss_sum / elements, global_step


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split: str,
    checkpoint: str,
    *,
    debug_numerics: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> dict:
    model.eval()
    metrics = MARSMetricAccumulator()
    batches = tqdm(
        loader, desc=progress_desc or split.capitalize(),
        leave=False, dynamic_ncols=True, disable=not show_progress,
    )
    for batch_index, (features, target) in enumerate(batches, start=1):
        prediction = model(features.to(device))
        if prediction.shape != target.shape:
            raise ValueError(f"{split} batch={batch_index}: prediction shape mismatch")
        if debug_numerics and not torch.isfinite(prediction).all():
            raise FloatingPointError(f"{split} batch={batch_index}: prediction is nonfinite")
        prediction_cpu = prediction.detach().cpu()
        target_cpu = target.cpu()
        try:
            metrics.update(prediction_cpu, target_cpu, debug_numerics=debug_numerics)
        except FloatingPointError as error:
            raise FloatingPointError(f"{split} batch={batch_index}: {error}") from error
    return {
        "split": split,
        "samples": metrics.sample_count,
        "checkpoint": checkpoint,
        "baseline_id": BASELINE_ID,
        **metrics.compute(),
    }
