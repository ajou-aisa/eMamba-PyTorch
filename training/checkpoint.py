import random
import subprocess
import sys
from pathlib import Path
from typing import Final

import numpy as np
import torch

from models.emamba import EMamba, NonlinearPolicy
from models.mamba.block import EMambaBlock


BASELINE_ID = "provisional_fp32_v2_flatten"
MODEL_DEFAULTS = {
    "d_model": 20, "expand": 2, "patch_size": 2, "num_blocks": 2,
    "d_state": 8, "out_dim": 57, "in_channels": 5,
}
ROOT = Path(__file__).resolve().parents[1]
_LAST_NATIVE_COMMIT: Final = "007ed66d386d366c90322d88288ecfb7e6fcbeee"
_FIRST_PIECEWISE_COMMIT: Final = "b8c091a0b16347a94024b322682dfa0034e43565"


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


def _legacy_nonlinear_policy(payload: dict) -> NonlinearPolicy | None:
    provenance = payload.get("git")
    if not isinstance(provenance, dict) or provenance.get("dirty") is not False:
        return None
    commit = provenance.get("commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        return None
    if commit == _FIRST_PIECEWISE_COMMIT:
        return "piecewise_fp32"
    if commit == _LAST_NATIVE_COMMIT:
        return "native_fp32"
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, _LAST_NATIVE_COMMIT),
        cwd=ROOT, capture_output=True, check=False,
    )
    return "native_fp32" if ancestor.returncode == 0 else None


def save_checkpoint(
    path: Path, model: EMamba, optimizer: torch.optim.Optimizer,
    epoch: int, global_step: int, best_rmse_cm: float,
    model_config: dict, training_config: dict, device: torch.device,
    *, rng_state: dict | None = None, scheduler_state: dict | None = None,
    scheduler_epoch_offset: int = 0,
) -> None:
    parameter_count, parameter_bytes = parameter_size(model)
    git_status = git_value("status", "--porcelain")
    payload = {
        "baseline_id": BASELINE_ID,
        "nonlinear_policy": model.nonlinear_policy,
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
    if scheduler_state is not None:
        payload["scheduler_state_dict"] = scheduler_state
        payload["scheduler_epoch_offset"] = scheduler_epoch_offset
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path, device: torch.device, *, legacy_nonlinear: NonlinearPolicy | None = None,
) -> tuple[EMamba, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("baseline_id") != BASELINE_ID:
        raise ValueError(f"checkpoint baseline_id must be {BASELINE_ID}")
    model_config = payload.get("model_config")
    if not isinstance(model_config, dict) or set(model_config) != set(MODEL_DEFAULTS) | {"readout"}:
        raise ValueError("checkpoint model_config is incomplete or unknown")
    if legacy_nonlinear is not None and legacy_nonlinear not in (
        "native_fp32", "piecewise_fp32"
    ):
        raise ValueError("legacy nonlinear policy override is invalid")
    if "nonlinear_policy" in payload:
        policy = payload["nonlinear_policy"]
        if policy not in ("native_fp32", "piecewise_fp32"):
            raise ValueError("checkpoint nonlinear policy is invalid")
        if legacy_nonlinear is not None and legacy_nonlinear != policy:
            raise ValueError("legacy nonlinear policy conflicts with checkpoint")
    else:
        policy = legacy_nonlinear or _legacy_nonlinear_policy(payload)
        if policy is None:
            raise ValueError(
                "checkpoint has ambiguous nonlinear policy; pass legacy_nonlinear="
                "'native_fp32' or 'piecewise_fp32'"
            )
    model = EMamba(**model_config, nonlinear_policy=policy)
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
