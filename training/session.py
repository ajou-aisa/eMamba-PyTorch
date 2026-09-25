import argparse
import math
import warnings
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.mars import build_dataloaders, training_data_fingerprint
from models.emamba import EMamba
from training.checkpoint import (
    MODEL_DEFAULTS, load_checkpoint, restore_optimizer_state, restore_rng_state,
)
from training.engine import GRADIENT_CLIP
from training.resume import STABLE_TRAINING_FIELDS, validate_resume_files, validate_training_config
from training.runtime import set_seed


@dataclass(frozen=True, slots=True)
class TrainingSession:
    model: EMamba
    loaders: dict[str, DataLoader]
    criterion: nn.MSELoss
    optimizer: torch.optim.Optimizer
    model_config: dict
    training_config: dict
    resume_payload: dict | None
    start_epoch: int
    global_step: int
    best_rmse_cm: float
    best_epoch: int
    best_validation: dict | None


def create_training_session(
    args: argparse.Namespace, device: torch.device, precision: dict[str, str],
) -> TrainingSession:
    resume_payload = None
    if args.resume is None:
        set_seed(args.seed)
        splits = ("train", "validation") if args.mode == "train" else ("train",)
        loaders = build_dataloaders(args.data_root, splits, args.batch_size, args.num_workers)
        data_fingerprint = training_data_fingerprint(args.data_root)
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
            "training_data_sha256": data_fingerprint,
            "tf32": precision,
        }
        args.output_dir.mkdir(parents=True, exist_ok=False)
        start_epoch = 1
        global_step = 0
        best_rmse_cm = float("inf")
        best_epoch = 0
        best_validation = None
    else:
        model, resume_payload = load_checkpoint(args.resume, device,
                                                legacy_nonlinear=args.legacy_nonlinear)
        saved_config = validate_training_config(resume_payload)
        saved_fingerprint = saved_config.get("training_data_sha256")
        if saved_fingerprint is None:
            warnings.warn("checkpoint lacks training data fingerprint; data identity is unverified",
                          RuntimeWarning, stacklevel=1)
        elif training_data_fingerprint(args.data_root) != saved_fingerprint:
            raise ValueError("resume training data differs from checkpoint")
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
        best_path = args.output_dir / "best.pt"
        if not best_path.is_file():
            raise ValueError("resume experiment is missing best.pt")
        best_model, best_payload = load_checkpoint(
            best_path, torch.device("cpu"), legacy_nonlinear=args.legacy_nonlinear,
        )
        best_config = validate_training_config(best_payload)
        if (best_payload.get("epoch") != best_epoch
                or best_payload.get("best_validation_rmse_cm") != resume_payload["best_validation_rmse_cm"]
                or best_payload.get("model_config") != resume_payload["model_config"]
                or best_model.nonlinear_policy != model.nonlinear_policy
                or best_config.get("training_data_sha256")
                != saved_config.get("training_data_sha256")
                or any(best_config[field] != saved_config[field]
                       for field in STABLE_TRAINING_FIELDS)):
            raise ValueError("best.pt conflicts with resume history or last.pt")
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
        optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[
            saved_config["optimizer"]
        ]
        optimizer = optimizer_class(
            model.parameters(), lr=args.lr, betas=tuple(saved_config["betas"]),
            weight_decay=saved_config["weight_decay"],
        )
        restore_optimizer_state(optimizer, resume_payload, device)
        for group in optimizer.param_groups:
            if (group.get("initial_lr", group["lr"]) != args.lr
                    or not math.isfinite(group["lr"]) or group["lr"] < 0
                    or tuple(group["betas"]) != tuple(saved_config["betas"])
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
    return TrainingSession(
        model, loaders, criterion, optimizer, model_config, training_config,
        resume_payload, start_epoch, global_step, best_rmse_cm, best_epoch, best_validation,
    )
