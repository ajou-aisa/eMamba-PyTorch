import argparse
from pathlib import Path

from training.checkpoint import BASELINE_ID
from training.workflow import run


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
    run(parse_args())


if __name__ == "__main__":
    main()
