import random
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


def capture_rng_state() -> dict:
    numpy_state = np.random.get_state(legacy=True)
    if isinstance(numpy_state, dict):
        raise RuntimeError("NumPy legacy RNG state is unavailable")
    mps_supported = (
        torch.backends.mps.is_available()
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
    )
    return {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0], numpy_state[1].tolist(), numpy_state[2],
            numpy_state[3], numpy_state[4],
        ),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if mps_supported else None,
        "device_rng_support": {
            "cuda": torch.cuda.is_available(), "mps": mps_supported,
        },
    }


def restore_rng_state(state: dict, device: torch.device) -> None:
    if not isinstance(state, dict) or not all(
        key in state for key in ("python", "numpy", "torch", "device_rng_support")
    ):
        raise ValueError("checkpoint RNG state is missing or incomplete")
    numpy_state = state["numpy"]
    random.setstate(state["python"])
    np.random.set_state((
        numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
        numpy_state[2], numpy_state[3], numpy_state[4],
    ))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if device.type == "mps" and state.get("mps") is not None:
        torch.mps.set_rng_state(state["mps"])


def restore_optimizer_state(
    optimizer: torch.optim.Optimizer, payload: dict, device: torch.device,
) -> None:
    state_dict = payload.get("optimizer_state_dict")
    if (not isinstance(state_dict, dict)
            or not isinstance(state_dict.get("state"), dict)
            or not isinstance(state_dict.get("param_groups"), list)):
        raise ValueError("checkpoint optimizer state is missing or invalid")
    try:
        optimizer.load_state_dict(state_dict)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint optimizer state is invalid") from error
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


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
    *, rng_state: dict | None = None,
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
    if rng_state is not None:
        payload["rng_state"] = rng_state
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
