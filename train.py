import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import torch

from training.checkpoint import (
    BASELINE_ID, capture_rng_state, delta_configuration, load_checkpoint,
    parameter_size, save_checkpoint,
)
from training.diagnostics import diagnose_delta
from training.data import SPLIT_FILES as SPLIT_FILES, SOURCE_SIZES as SOURCE_SIZES
from training.data import SPLIT_SIZES, build_dataloaders
from training.engine import evaluate, train_one_epoch
from training.reporting import (
    print_epoch_header, print_epoch_result, print_eval_summary,
    print_resume_summary, print_run_summary, print_smoke_summary,
    print_training_summary,
)
from training.runtime import configure_fp32, select_device, set_seed as set_seed
from training.session import create_training_session


ROOT = Path(__file__).resolve().parent


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
    parser.add_argument("--readout", choices=("flatten",))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--debug-numerics", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--json-stdout", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--legacy-nonlinear", choices=("native_fp32", "piecewise_fp32"))
    parser.add_argument("--split", choices=("validation", "test"))
    args = parser.parse_args()
    if args.legacy_nonlinear is not None and args.resume is None and args.mode != "eval":
        parser.error("--legacy-nonlinear requires --resume or --mode eval")
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
            args.output_dir = args.output_dir or ROOT / "results" / f"{BASELINE_ID}_smoke"
        case "train":
            if args.steps is not None or args.checkpoint or args.split:
                parser.error("train does not use --steps, --checkpoint, or --split")
            args.epochs = 1 if args.epochs is None else args.epochs
            args.output_dir = args.output_dir or ROOT / "results" / BASELINE_ID
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
            args.readout = args.readout or "flatten"
    if args.resume is None:
        args.batch_size = 128 if args.batch_size is None else args.batch_size
        args.num_workers = 0 if args.num_workers is None else args.num_workers
    return args


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    precision = configure_fp32(device)
    show_progress = not args.no_progress and not args.json_stdout and sys.stderr.isatty()
    if args.mode == "eval":
        model, _ = load_checkpoint(args.checkpoint, device,
                                   legacy_nonlinear=args.legacy_nonlinear)
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

    session = create_training_session(args, device, precision)
    model = session.model
    loaders = session.loaders
    criterion = session.criterion
    optimizer = session.optimizer
    model_config = session.model_config
    training_config = session.training_config
    resume_payload = session.resume_payload
    start_epoch = session.start_epoch
    global_step = session.global_step
    best_rmse_cm = session.best_rmse_cm
    best_epoch = session.best_epoch
    best_validation = session.best_validation
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
    scheduler_epoch_offset = 0
    if resume_payload is not None:
        scheduler_state = resume_payload.get("scheduler_state_dict")
        saved_offset = 0
        if scheduler_state is not None:
            saved_offset = resume_payload.get("scheduler_epoch_offset", 0)
            if (not isinstance(scheduler_state, dict)
                    or type(scheduler_state.get("last_epoch")) is not int
                    or scheduler_state["last_epoch"] < 0
                    or type(saved_offset) is not int or saved_offset < 0
                    or scheduler_state["last_epoch"] + saved_offset != resume_payload["epoch"]
                    or scheduler_state.get("T_max") != resume_payload["training_config"]["epochs"]
                    or scheduler_state.get("eta_min") != 1e-6
                    or scheduler_state.get("_last_lr")
                    != [group["lr"] for group in optimizer.param_groups]):
                raise ValueError("checkpoint scheduler state conflicts with optimizer or epoch")
        if (scheduler_state is not None
                and resume_payload["training_config"]["epochs"] == args.epochs):
            scheduler.load_state_dict(scheduler_state)
            scheduler_epoch_offset = saved_offset
        else:
            scheduler_epoch_offset = resume_payload["epoch"]
            warnings.warn("scheduler restarts because its state or original epoch target is unavailable",
                          RuntimeWarning, stacklevel=1)

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
                scheduler_state=scheduler.state_dict(),
                scheduler_epoch_offset=scheduler_epoch_offset,
            )
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, epoch, global_step,
            best_rmse_cm, model_config, training_config, device, rng_state=rng_state,
            scheduler_state=scheduler.state_dict(),
            scheduler_epoch_offset=scheduler_epoch_offset,
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
