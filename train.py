import argparse
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, random_split

from datasets.mars import MARSDataset
from models.emamba import EMamba
from training.checkpoint import (
    BASELINE_ID, MODEL_DEFAULTS, capture_rng_state, delta_configuration,
    load_checkpoint, parameter_size, restore_optimizer_state, restore_rng_state,
    save_checkpoint,
)
from training.diagnostics import diagnose_delta
from training.engine import GRADIENT_CLIP, evaluate, train_one_epoch
from training.reporting import (
    print_epoch_header, print_epoch_result, print_eval_summary,
    print_resume_summary, print_run_summary, print_smoke_summary,
    print_training_summary,
)
from training.resume import validate_resume_files, validate_training_config


ROOT = Path(__file__).resolve().parent
SPLIT_FILES = {"train": "train", "validation": "validate", "test": "test"}
SOURCE_SIZES: Final = {"train": 24066, "validation": 8033, "test": 7984}
SPLIT_SIZES: Final = {"train": 25652, "validation": 6414, "test": 8017}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Provisional FP32 training on MARS")
    parser.add_argument("--mode", required=True, choices=("smoke", "train", "eval"))
    parser.add_argument("--data-root", type=Path, default=ROOT / "third_party/MARS/feature")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--readout", choices=("mean", "last"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--debug-numerics", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-stdout", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--split", choices=("validation", "test"))
    args = parser.parse_args()
    if args.resume is not None:
        if args.mode != "train":
            parser.error("--resume is only valid with --mode train")
        if args.resume.name != "last.pt":
            parser.error("--resume must use last.pt; best.pt is for evaluation")
        if args.epochs is None:
            parser.error("--epochs is required with --resume")
        if any(value is not None for value in
               (args.lr, args.seed, args.readout, args.batch_size, args.num_workers)):
            parser.error("resume uses checkpoint hyperparameters; CLI overrides are invalid")
        resume_dir = args.resume.resolve().parent
        if args.output_dir is not None and args.output_dir.resolve() != resume_dir:
            parser.error("--output-dir must match the resume checkpoint directory")
        args.output_dir = resume_dir
    if (args.batch_size is not None and args.batch_size <= 0) or (
        args.num_workers is not None and args.num_workers < 0
    ):
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
        if args.resume is None:
            args.lr = 1e-3 if args.lr is None else args.lr
            args.seed = 0 if args.seed is None else args.seed
            args.readout = args.readout or "mean"
    if args.resume is None:
        args.batch_size = 128 if args.batch_size is None else args.batch_size
        args.num_workers = 0 if args.num_workers is None else args.num_workers
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
    sources = []
    for split, suffix in SPLIT_FILES.items():
        features = data_root / f"featuremap_{suffix}.npy"
        labels = data_root / f"labels_{suffix}.npy"
        dataset = MARSDataset(features, labels)
        if len(dataset) != SOURCE_SIZES[split]:
            raise ValueError(
                f"{split} at {data_root}: expected {SOURCE_SIZES[split]} samples, "
                f"got {len(dataset)}"
            )
        sources.append(dataset)
    partitions = random_split(
        ConcatDataset(sources), list(SPLIT_SIZES.values()),
        generator=torch.Generator().manual_seed(0),
    )
    datasets = dict(zip(SPLIT_SIZES, partitions, strict=True))
    loaders = {}
    for split in splits:
        loaders[split] = DataLoader(
            datasets[split], batch_size=batch_size, shuffle=split == "train",
            num_workers=num_workers, drop_last=False,
        )
    return loaders


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    precision = configure_fp32(device)
    show_progress = not args.no_progress and not args.json_stdout and sys.stderr.isatty()
    if args.mode == "eval":
        model, _ = load_checkpoint(args.checkpoint, device)
        loaders = build_dataloaders(
            args.data_root, (args.split,), args.batch_size, args.num_workers,
        )
        result = evaluate(
            model, loaders[args.split], device, args.split, str(args.checkpoint),
            debug_numerics=args.debug_numerics,
            show_progress=show_progress, progress_desc="Validate",
        )
        if args.json_stdout:
            print(json.dumps({"mode": "eval", "precision": precision, **result},
                             sort_keys=True, allow_nan=False))
        else:
            print_eval_summary(result, str(device))
        return

    resume_payload = None
    if args.resume is None:
        set_seed(args.seed)
        splits = ("train", "validation") if args.mode == "train" else ("train",)
        loaders = build_dataloaders(args.data_root, splits, args.batch_size, args.num_workers)
        model_config = {**MODEL_DEFAULTS, "readout": args.readout}
        model = EMamba(**model_config).to(device).float()
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
        )
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
        start_epoch = 1
        global_step = 0
        best_rmse_cm = float("inf")
        best_epoch = 0
        best_validation = None
    else:
        model, resume_payload = load_checkpoint(args.resume, device)
        saved_config = validate_training_config(resume_payload)
        checkpoint_epoch = resume_payload["epoch"]
        if type(checkpoint_epoch) is not int or checkpoint_epoch < 1:
            raise ValueError("resume checkpoint epoch is invalid")
        if args.epochs <= checkpoint_epoch:
            raise ValueError(
                f"resume checkpoint is already at epoch {checkpoint_epoch}, "
                f"but target --epochs is {args.epochs}"
            )
        best_epoch, best_validation, config_present = validate_resume_files(
            args.output_dir, resume_payload,
        )
        if not (args.output_dir / "best.pt").exists():
            raise ValueError("resume experiment is missing best.pt")
        if not config_present:
            warnings.warn("run_config.json is missing; using checkpoint and history only",
                          RuntimeWarning, stacklevel=1)
        args.lr = saved_config["lr"]
        args.batch_size = saved_config["batch_size"]
        args.seed = saved_config["seed"]
        args.readout = resume_payload["model_config"]["readout"]
        args.num_workers = saved_config["num_workers"]
        model_config = resume_payload["model_config"]
        loaders = build_dataloaders(
            args.data_root, ("train", "validation"), args.batch_size, args.num_workers,
        )
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=tuple(saved_config["betas"]),
            weight_decay=saved_config["weight_decay"],
        )
        restore_optimizer_state(optimizer, resume_payload, device)
        for group in optimizer.param_groups:
            if (group["lr"] != args.lr or tuple(group["betas"]) != tuple(saved_config["betas"])
                    or group["weight_decay"] != saved_config["weight_decay"]):
                raise ValueError("checkpoint optimizer state conflicts with training_config")
        training_config = {**saved_config, "epochs": args.epochs, "tf32": precision}
        rng_state = resume_payload.get("rng_state")
        if rng_state is None:
            warnings.warn(
                "checkpoint lacks RNG state; reseeding from saved seed, "
                "so continuation is not bitwise reproducible",
                RuntimeWarning, stacklevel=1,
            )
            set_seed(args.seed)
        else:
            restore_rng_state(rng_state, device)
            if device.type in ("cuda", "mps") and rng_state.get(device.type) is None:
                warnings.warn(
                    f"checkpoint lacks {device.type} RNG state; exact device continuation "
                    "is not guaranteed", RuntimeWarning, stacklevel=1,
                )
        start_epoch = checkpoint_epoch + 1
        global_step = resume_payload["global_step"]
        best_rmse_cm = resume_payload["best_validation_rmse_cm"]
    parameter_count, parameter_bytes = parameter_size(model)
    run_config = {
        "mode": args.mode, "baseline_id": BASELINE_ID, "device": str(device),
        "precision": precision, "model_config": model_config,
        "delta_config": delta_configuration(model),
        "parameter_count": parameter_count, "fp32_parameter_bytes": parameter_bytes,
        "training_config": training_config,
    }
    if args.resume is None:
        (args.output_dir / "run_config.json").write_text(
            json.dumps(run_config, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    if args.json_stdout:
        print(json.dumps(run_config, sort_keys=True, allow_nan=False))
    elif resume_payload is not None:
        print_resume_summary(args.resume, resume_payload, args.epochs, str(device))
    elif args.mode == "train":
        sample_counts = {split: SPLIT_SIZES[split] for split in loaders}
        print_run_summary(run_config, args.output_dir, sample_counts)

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
        if args.json_stdout:
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        else:
            print_smoke_summary(run_config, result)
        return

    last_validation = None
    history_path = args.output_dir / "history.jsonl"
    if not args.json_stdout:
        print_epoch_header()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, global_step = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, epoch, global_step,
            debug_numerics=args.debug_numerics,
            show_progress=show_progress, progress_desc=f"Epoch {epoch}/{args.epochs}",
        )
        validation = evaluate(
            model, loaders["validation"], device, "validation", "current",
            debug_numerics=args.debug_numerics,
            show_progress=show_progress, progress_desc="Validate",
        )
        last_validation = validation
        current_rmse_cm = validation["rmse_cm"]["all"]
        if not math.isfinite(current_rmse_cm):
            raise FloatingPointError(f"validation epoch={epoch}: mean RMSE is nonfinite")
        is_best = current_rmse_cm < best_rmse_cm
        scheduler.step()
        rng_state = capture_rng_state()
        if is_best:
            best_rmse_cm = current_rmse_cm
            best_epoch = epoch
            best_validation = validation
            save_checkpoint(
                args.output_dir / "best.pt", model, optimizer, epoch, global_step,
                best_rmse_cm, model_config, training_config, device, rng_state=rng_state,
            )
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, epoch, global_step,
            best_rmse_cm, model_config, training_config, device, rng_state=rng_state,
        )
        record = {"epoch": epoch, "global_step": global_step, "train_loss": train_loss,
                  "validation": validation, "best_validation_rmse_cm": best_rmse_cm}
        with history_path.open("a") as history:
            history.write(json.dumps(record, allow_nan=False) + "\n")
        if args.json_stdout:
            print(json.dumps(record, sort_keys=True, allow_nan=False))
        else:
            print_epoch_result(
                epoch, args.epochs, train_loss, validation, best_rmse_cm, is_best,
                show_axis_metrics=args.debug_numerics,
            )

        if epoch - best_epoch >= 15:
            break

    if not args.json_stdout:
        assert best_validation is not None and last_validation is not None
        print_training_summary(
            best_epoch, best_validation, last_validation, args.output_dir / "best.pt",
        )


if __name__ == "__main__":
    main()
