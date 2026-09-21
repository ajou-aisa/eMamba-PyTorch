import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.mars import MARSDataset
from models.emamba import EMamba
from training.checkpoint import (
    BASELINE_ID, MODEL_DEFAULTS, delta_configuration, load_checkpoint,
    parameter_size, save_checkpoint,
)
from training.diagnostics import diagnose_delta
from training.engine import GRADIENT_CLIP, evaluate, train_one_epoch


ROOT = Path(__file__).resolve().parent
SPLIT_FILES = {"train": "train", "validation": "validate", "test": "test"}
SPLIT_SIZES = {"train": 24066, "validation": 8033, "test": 7984}


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
    parser.add_argument("--debug-numerics", action="store_true")
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


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    precision = configure_fp32(device)
    if args.mode == "eval":
        model, _ = load_checkpoint(args.checkpoint, device)
        loaders = build_dataloaders(
            args.data_root, (args.split,), args.batch_size, args.num_workers,
        )
        result = evaluate(
            model, loaders[args.split], device, args.split, str(args.checkpoint),
            debug_numerics=args.debug_numerics,
        )
        print(json.dumps({"mode": "eval", "precision": precision, **result},
                         sort_keys=True, allow_nan=False))
        return

    set_seed(args.seed)
    splits = ("train", "validation") if args.mode == "train" else ("train",)
    loaders = build_dataloaders(args.data_root, splits, args.batch_size, args.num_workers)
    model_config = {**MODEL_DEFAULTS, "readout": args.readout}
    model = EMamba(**model_config).to(device).float()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0,
    )
    parameter_count, parameter_bytes = parameter_size(model)
    training_config = {
        "precision": "fp32", "optimizer": type(optimizer).__name__,
        "lr": optimizer.defaults["lr"], "betas": optimizer.defaults["betas"],
        "weight_decay": optimizer.defaults["weight_decay"], "loss": type(criterion).__name__,
        "batch_size": args.batch_size, "gradient_clip": GRADIENT_CLIP,
        "seed": args.seed, "num_workers": args.num_workers,
        "epochs": args.epochs if args.mode == "train" else None,
        "steps": args.steps if args.mode == "smoke" else None,
        "tf32": precision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(json.dumps({
        "mode": args.mode, "baseline_id": BASELINE_ID, "device": str(device),
        "precision": precision, "model_config": model_config,
        "delta_config": delta_configuration(model),
        "parameter_count": parameter_count, "fp32_parameter_bytes": parameter_bytes,
        "training_config": training_config,
    }, sort_keys=True, allow_nan=False))

    if args.mode == "smoke":
        fixed_batch = next(iter(loaders["train"]))
        frames, target = (item.to(device) for item in fixed_batch)
        with torch.no_grad():
            initial_loss = criterion(model(frames), target).item()
        if not math.isfinite(initial_loss):
            raise FloatingPointError("smoke initial loss is nonfinite")
        initial_delta = diagnose_delta(model, frames)
        global_step = 0
        for _ in range(args.steps):
            _, global_step = train_one_epoch(
                model, [fixed_batch], criterion, optimizer, device, 1, global_step,
                debug_numerics=True,
            )
        model.eval()
        with torch.no_grad():
            final_loss = criterion(model(frames), target).item()
        if not math.isfinite(final_loss):
            raise FloatingPointError("smoke final loss is nonfinite")
        result = {
            "mode": "smoke", "baseline_id": BASELINE_ID, "steps": global_step,
            "initial_loss": initial_loss, "final_loss": final_loss,
            "finite": True,
            "delta_initial": initial_delta, "delta_final": diagnose_delta(model, frames),
        }
        (args.output_dir / "smoke.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return

    best_rmse_cm = float("inf")
    global_step = 0
    history_path = args.output_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch, global_step,
            debug_numerics=args.debug_numerics,
        )
        validation = evaluate(
            model, loaders["validation"], device, "validation", "current",
            debug_numerics=args.debug_numerics,
        )
        current_rmse_cm = validation["rmse_cm"]["all"]
        if not math.isfinite(current_rmse_cm):
            raise FloatingPointError(f"validation epoch={epoch}: mean RMSE is nonfinite")
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
            history.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
