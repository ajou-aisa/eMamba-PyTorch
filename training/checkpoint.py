import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from models.emamba import EMamba
from models.mamba.block import EMambaBlock


BASELINE_ID = "provisional_fp32_v1"
MODEL_DEFAULTS = {
    "d_model": 20, "expand": 2, "patch_size": 2, "num_blocks": 2,
    "d_state": 8, "out_dim": 57, "in_channels": 5,
}
ROOT = Path(__file__).resolve().parents[1]


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


def parameter_size(model: EMamba) -> tuple[int, int]:
    parameters = tuple(model.parameters())
    return (
        sum(parameter.numel() for parameter in parameters),
        sum(parameter.numel() * parameter.element_size() for parameter in parameters),
    )


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
    parameter_count, parameter_bytes = parameter_size(model)
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
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("baseline_id") != BASELINE_ID:
        raise ValueError(f"checkpoint baseline_id must be {BASELINE_ID}")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or set(model_config) != set(MODEL_DEFAULTS) | {"readout"}:
        raise ValueError("checkpoint model_config is incomplete or unknown")
    model = EMamba(**model_config)
    if payload.get("readout") != model.head.readout:
        raise ValueError("checkpoint readout conflicts with model_config")
    if payload.get("delta_config") != delta_configuration(model):
        raise ValueError("checkpoint delta configuration conflicts with this baseline")
    parameter_count, parameter_bytes = parameter_size(model)
    if (payload.get("parameter_count") != parameter_count
            or payload.get("fp32_parameter_bytes") != parameter_bytes):
        raise ValueError("checkpoint parameter count conflicts with model_config")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).float(), payload
