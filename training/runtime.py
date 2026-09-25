import random

import numpy as np
import torch


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
