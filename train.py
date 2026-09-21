import argparse
import json
import random
import subprocess
import sys
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from datasets.mars import MARSDataset
from models.emamba import EMamba
from models.mamba.block import EMambaBlock
from models.mamba.selective_ssm import SelectiveSSM


BASELINE_ID = "provisional_fp32_v1"
GRADIENT_CLIP = 1.0
ROOT = Path(__file__).resolve().parent
SPLIT_FILES = {"train": "train", "validation": "validate", "test": "test"}
SPLIT_SIZES = {"train": 24066, "validation": 8033, "test": 7984}
MODEL_DEFAULTS = {
    "d_model": 20, "expand": 2, "patch_size": 2, "num_blocks": 2,
    "d_state": 8, "out_dim": 57, "in_channels": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provisional FP32 training on MARS")
    parser.add_argument("--mode", required=True, choices=("smoke", "train", "eval"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "third_party/MARS/feature")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--readout", choices=("mean", "last"))
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"))
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        parser.error("--batch-size must be positive and --num-workers nonnegative")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr must be positive")
    match args.mode:
        case "smoke":
            if args.epochs is not None or args.checkpoint or args.split:
                parser.error("smoke does not use --epochs, --checkpoint, or --split")
            args.steps = 25 if args.steps is None else args.steps
            args.output_dir = args.output_dir or ROOT / "results/provisional_fp32_v1_smoke"
        case "train":
            if args.steps is not None or args.checkpoint or args.split:
                parser.error("train does not use --steps, --checkpoint, or --split")
            args.epochs = 1 if args.epochs is None else args.epochs
            args.output_dir = args.output_dir or ROOT / "results/provisional_fp32_v1"
        case "eval":
            if args.checkpoint is None:
                parser.error("eval requires --checkpoint")
            if any(value is not None for value in
                   (args.epochs, args.steps, args.lr, args.seed, args.readout, args.output_dir)):
                parser.error("eval uses checkpoint settings; training options are invalid")
            args.split = args.split or "validation"
    if args.mode != "eval":
        args.lr = 1e-3 if args.lr is None else args.lr
        args.seed = 0 if args.seed is None else args.seed
        args.readout = args.readout or "mean"
    return args


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(requested)


def configure_fp32(device: torch.device) -> dict[str, str]:
    if device.type != "cuda":
        return {"tf32": "not_applicable"}
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
        return {
            "api": "fp32_precision",
            "matmul": torch.backends.cuda.matmul.fp32_precision,
            "convolution": torch.backends.cudnn.fp32_precision,
        }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {"api": "allow_tf32", "matmul": "false", "convolution": "false"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    data_root: Path, splits: tuple[str, ...], batch_size: int, num_workers: int,
) -> dict[str, DataLoader]:
    loaders = {}
    for split in splits:
        suffix = SPLIT_FILES[split]
        features = data_root / f"featuremap_{suffix}.npy"
        labels = data_root / f"labels_{suffix}.npy"
        dataset = MARSDataset(features, labels)
        if len(dataset) != SPLIT_SIZES[split]:
            raise ValueError(
                f"{split} at {data_root}: expected {SPLIT_SIZES[split]} samples, "
                f"got {len(dataset)}"
            )
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=split == "train",
            num_workers=num_workers, drop_last=False,
        )
    return loaders


def train_one_epoch(
    model: nn.Module, loader: Iterable[tuple[Tensor, Tensor]], optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, global_step: int,
) -> tuple[float, int]:
    model.train()
    loss_function = nn.MSELoss()
    loss_sum = 0.0
    elements = 0
    for batch_index, (features, target) in enumerate(loader, start=1):
        features, target = features.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        location = f"epoch={epoch} batch={batch_index} step={global_step + 1}"
        if prediction.shape != target.shape:
            raise ValueError(f"{location} prediction shape {prediction.shape} != {target.shape}")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"{location} prediction is nonfinite")
        loss = loss_function(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{location} loss is nonfinite")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise FloatingPointError(f"{location} gradient is nonfinite")
        try:
            clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP, error_if_nonfinite=True)
        except RuntimeError as error:
            raise FloatingPointError(f"{location} gradient norm is nonfinite") from error
        optimizer.step()
        loss_sum += loss.item() * target.numel()
        elements += target.numel()
        global_step += 1
    if elements == 0:
        raise ValueError(f"epoch={epoch} has no training samples")
    return loss_sum / elements, global_step


def compute_mars_metrics(
    abs_error_sum: np.ndarray, squared_error_sum: np.ndarray, sample_count: int,
) -> dict[str, dict[str, float]]:
    if sample_count <= 0:
        raise ValueError("MARS metric requires at least one sample")
    if abs_error_sum.shape != (57,) or squared_error_sum.shape != (57,):
        raise ValueError("MARS metric requires 57 coordinate sums")
    per_coordinate = {
        "mae_cm": abs_error_sum / sample_count * 100,
        "rmse_cm": np.sqrt(squared_error_sum / sample_count) * 100,
    }
    return {
        name: {
            "x": float(values[0:19].mean()),
            "y": float(values[19:38].mean()),
            "z": float(values[38:57].mean()),
            "all": float(values.mean()),
        }
        for name, values in per_coordinate.items()
    }


@torch.no_grad()
def evaluate(
    model: EMamba, loader: DataLoader, device: torch.device,
    split: str, checkpoint: str,
) -> dict:
    model.eval()
    abs_error_sum = np.zeros(57, dtype=np.float64)
    squared_error_sum = np.zeros(57, dtype=np.float64)
    sample_count = 0
    for batch_index, (features, target) in enumerate(loader, start=1):
        prediction = model(features.to(device))
        if prediction.shape != target.shape:
            raise ValueError(f"{split} batch={batch_index}: prediction shape mismatch")
        if not torch.isfinite(prediction).all():
            raise FloatingPointError(f"{split} batch={batch_index}: prediction is nonfinite")
        error = prediction.detach().to("cpu", dtype=torch.float64) - target.to(torch.float64)
        abs_error_sum += error.abs().sum(dim=0).numpy()
        squared_error_sum += error.square().sum(dim=0).numpy()
        sample_count += target.shape[0]
    return {
        "split": split, "samples": sample_count, "checkpoint": checkpoint,
        "baseline_id": BASELINE_ID,
        **compute_mars_metrics(abs_error_sum, squared_error_sum, sample_count),
    }


def delta_configuration(model: EMamba) -> dict[str, str | float]:
    block = model.blocks[0]
    assert isinstance(block, EMambaBlock)
    ssm = block.ssm
    return {
        "delta_activation": ssm.delta_activation,
        "dt_init_min": ssm.dt_init_min,
        "dt_init_max": ssm.dt_init_max,
        "dt_init_scale": ssm.dt_init_scale,
    }


def git_value(*args: str) -> str | None:
    process = subprocess.run(
        ("git", *args), cwd=ROOT, capture_output=True, text=True, check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


def save_checkpoint(
    path: Path, model: EMamba, optimizer: torch.optim.Optimizer,
    epoch: int, global_step: int, best_rmse_cm: float,
    model_config: dict, training_config: dict, device: torch.device,
) -> None:
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in model.parameters())
    git_status = git_value("status", "--porcelain")
    payload = {
        "baseline_id": BASELINE_ID,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch, "global_step": global_step,
        "best_validation_rmse_cm": best_rmse_cm,
        "model_config": model_config, "readout": model.head.readout,
        "training_config": training_config,
        "delta_config": delta_configuration(model),
        "parameter_count": parameter_count,
        "fp32_parameter_bytes": parameter_bytes,
        "environment": {
            "python": sys.version.split()[0], "torch": str(torch.__version__),
            "numpy": np.__version__, "device": str(device),
        },
        "git": {
            "commit": git_value("rev-parse", "HEAD"),
            "dirty": None if git_status is None else bool(git_status),
            "mars_commit": git_value("-C", "third_party/MARS", "rev-parse", "HEAD"),
        },
    }
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[EMamba, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("baseline_id") != BASELINE_ID:
        raise ValueError(f"checkpoint baseline_id must be {BASELINE_ID}")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or set(model_config) != set(MODEL_DEFAULTS) | {"readout"}:
        raise ValueError("checkpoint model_config is incomplete or unknown")
    model = EMamba(**model_config).to(device).float()
    if payload.get("readout") != model.head.readout:
        raise ValueError("checkpoint readout conflicts with model_config")
    if payload.get("delta_config") != delta_configuration(model):
        raise ValueError("checkpoint delta configuration conflicts with this baseline")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in model.parameters())
    if (payload.get("parameter_count") != parameter_count
            or payload.get("fp32_parameter_bytes") != parameter_bytes):
        raise ValueError("checkpoint parameter count conflicts with model_config")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def diagnose_delta(model: EMamba, frames: Tensor) -> list[dict[str, float]]:
    stats = [{} for _ in model.blocks]
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


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    precision = configure_fp32(device)
    if args.mode == "eval":
        model, payload = load_checkpoint(args.checkpoint, device)
        loader = build_dataloaders(args.data_root, (args.split,), args.batch_size, args.num_workers)
        result = evaluate(model, loader[args.split], device, args.split, str(args.checkpoint))
        print(json.dumps({"mode": "eval", "precision": precision, **result}, sort_keys=True))
        return

    set_seed(args.seed)
    model_config = {**MODEL_DEFAULTS, "readout": args.readout}
    model = EMamba(**model_config).to(device).float()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size()
                          for parameter in model.parameters())
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0,
    )
    training_config = {
        "precision": "fp32", "optimizer": type(optimizer).__name__,
        "lr": optimizer.defaults["lr"], "betas": optimizer.defaults["betas"],
        "weight_decay": optimizer.defaults["weight_decay"], "loss": nn.MSELoss.__name__,
        "batch_size": args.batch_size, "gradient_clip": GRADIENT_CLIP,
        "seed": args.seed, "num_workers": args.num_workers,
        "epochs": args.epochs if args.mode == "train" else None,
        "steps": args.steps if args.mode == "smoke" else None,
        "tf32": precision,
    }
    splits = ("train", "validation") if args.mode == "train" else ("train",)
    loaders = build_dataloaders(args.data_root, splits, args.batch_size, args.num_workers)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(json.dumps({
        "mode": args.mode, "baseline_id": BASELINE_ID, "device": str(device),
        "precision": precision, "model_config": model_config,
        "delta_config": delta_configuration(model),
        "parameter_count": parameter_count, "fp32_parameter_bytes": parameter_bytes,
        "training_config": training_config,
    }, sort_keys=True))

    if args.mode == "smoke":
        fixed_batch = next(iter(loaders["train"]))
        frames, target = (item.to(device) for item in fixed_batch)
        with torch.no_grad():
            initial_loss = nn.functional.mse_loss(model(frames), target).item()
        initial_delta = diagnose_delta(model, frames)
        global_step = 0
        for step in range(args.steps):
            _, global_step = train_one_epoch(
                model, [fixed_batch], optimizer, device, 1, global_step,
            )
        model.eval()
        with torch.no_grad():
            final_loss = nn.functional.mse_loss(model(frames), target).item()
        result = {
            "mode": "smoke", "baseline_id": BASELINE_ID, "steps": global_step,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "finite": bool(np.isfinite(initial_loss) and np.isfinite(final_loss)),
            "delta_initial": initial_delta, "delta_final": diagnose_delta(model, frames),
        }
        (args.output_dir / "smoke.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True))
        return

    best_rmse_cm = float("inf")
    global_step = 0
    history_path = args.output_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, loaders["train"], optimizer, device, epoch, global_step,
        )
        validation = evaluate(model, loaders["validation"], device, "validation", "current")
        current_rmse_cm = validation["rmse_cm"]["all"]
        if current_rmse_cm < best_rmse_cm:
            best_rmse_cm = current_rmse_cm
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, epoch, global_step,
                best_rmse_cm, model_config, training_config, device,
            )
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, epoch, global_step,
            best_rmse_cm, model_config, training_config, device,
        )
        record = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss,
                  "validation": validation, "best_validation_rmse_cm": best_rmse_cm}
        with history_path.open("a") as history:
            history.write(json.dumps(record) + "\n")
        print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
