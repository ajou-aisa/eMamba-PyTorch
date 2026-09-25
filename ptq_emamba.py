import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence, assert_never

import torch

from models.emamba import NonlinearPolicy
from ptq.calibrate import profile_hash
from ptq.io import load_quantized
from ptq.report import ArtifactReport, WorkflowReport
from ptq.workflow import convert
from training.data import build_dataloaders
from training.engine import evaluate
from training.runtime import configure_fp32


Split = Literal["validation", "test"]
DeviceName = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True, slots=True)
class Arguments:
    checkpoint: Path | None
    artifact: Path | None
    output_dir: Path | None
    data_root: Path
    split: Split
    device: DeviceName
    batch_size: int
    piecewise: bool | None
    legacy_nonlinear: NonlinearPolicy | None


class CLIError(RuntimeError):
    pass

def parse_args(argv: Sequence[str] | None = None) -> Arguments:
    parser = argparse.ArgumentParser(description="Calibrate, export, or evaluate eMamba INT8 PTQ")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--artifact", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("third_party/MARS/feature"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    nonlinear = parser.add_mutually_exclusive_group()
    nonlinear.add_argument("--piecewise", dest="piecewise", action="store_const", const=True,
                           default=None, help="Use piecewise SiLU/exp for PTQ")
    nonlinear.add_argument("--native", dest="piecewise", action="store_const", const=False,
                           help="Use native SiLU/exp for PTQ")
    parser.add_argument("--legacy-nonlinear", choices=("native_fp32", "piecewise_fp32"),
                        help="Resolve an untagged checkpoint with ambiguous provenance")
    raw = parser.parse_args(argv)
    if raw.checkpoint is not None and raw.output_dir is None:
        parser.error("--checkpoint requires --output-dir")
    if raw.artifact is not None and raw.output_dir is not None:
        parser.error("--output-dir is only valid with --checkpoint")
    if raw.artifact is not None and (raw.piecewise is not None or raw.legacy_nonlinear is not None):
        parser.error("--artifact uses its saved nonlinear policy; omit nonlinear overrides")
    if raw.batch_size <= 0:
        parser.error("--batch-size must be positive")
    return Arguments(**vars(raw))


def select_device(requested: DeviceName) -> torch.device:
    match requested:
        case "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        case "cuda":
            if not torch.cuda.is_available():
                raise CLIError("CUDA was requested but is unavailable")
            return torch.device("cuda")
        case "cpu":
            return torch.device("cpu")
        case unreachable:
            assert_never(unreachable)


def run(arguments: Arguments) -> WorkflowReport | ArtifactReport:
    device = select_device(arguments.device)
    precision = configure_fp32(device)
    match arguments.checkpoint, arguments.artifact:
        case checkpoint, None if checkpoint is not None:
            if arguments.output_dir is None:
                raise CLIError("conversion output directory is missing")
            return convert(checkpoint, arguments.output_dir, arguments.data_root,
                           arguments.split, device, batch_size=arguments.batch_size,
                           precision=precision, use_pwl=arguments.piecewise,
                           legacy_nonlinear=arguments.legacy_nonlinear)
        case None, artifact if artifact is not None:
            loaded = load_quantized(artifact)
            initial_hash = profile_hash(loaded.profile)
            loader = build_dataloaders(
                arguments.data_root, (arguments.split,), arguments.batch_size, 0,
            )[arguments.split]
            metric = evaluate(loaded.model.to(device), loader, device, arguments.split,
                              str(artifact))
            final_hash = profile_hash(loaded.profile)
            return {"schema_version": 1, "mode": "artifact", "evaluation": metric,
                    "precision": precision, "profile_hash": final_hash,
                    "profile_unchanged": initial_hash == final_hash}
        case unreachable:
            assert_never(unreachable)


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
