import json
import math
from pathlib import Path


STABLE_TRAINING_FIELDS = (
    "optimizer", "loss", "precision", "lr", "betas", "weight_decay",
    "gradient_clip", "batch_size", "seed", "num_workers",
)


def validate_training_config(payload: dict) -> dict:
    config = payload.get("training_config")
    if not isinstance(config, dict) or any(
        field not in config for field in STABLE_TRAINING_FIELDS
    ):
        raise ValueError("checkpoint training_config is incomplete")
    if config["optimizer"] not in ("Adam", "AdamW"):
        raise ValueError("checkpoint training_config optimizer is unsupported")
    for field, expected in (
        ("loss", "MSELoss"),
        ("precision", "fp32"), ("gradient_clip", 1.0),
    ):
        if config[field] != expected:
            raise ValueError(f"checkpoint training_config {field} is unsupported")
    betas = config["betas"]
    if (not isinstance(betas, (tuple, list)) or len(betas) != 2
            or any(type(beta) not in (int, float) or not 0 <= beta < 1 for beta in betas)):
        raise ValueError("checkpoint training_config betas are invalid")
    for field in ("lr", "weight_decay"):
        value = config[field]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or (value <= 0 if field == "lr" else value < 0)):
            raise ValueError(f"checkpoint training_config {field} is invalid")
    for field, lower_bound in (("batch_size", 1), ("seed", 0), ("num_workers", 0)):
        value = config[field]
        if type(value) is not int or value < lower_bound:
            raise ValueError(f"checkpoint training_config {field} is invalid")
    return config


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON value {value}")


def _parse_finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"nonfinite JSON value {value}")
    return number


def _load_json(text: str, path: Path) -> dict:
    try:
        value = json.loads(
            text, parse_constant=_reject_nonfinite, parse_float=_parse_finite_float,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _finite_metric(value: float | int | None, name: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value)):
        raise ValueError(f"history.jsonl: {name} must be finite")
    return float(value)


def _validate_run_config(path: Path, payload: dict, training_config: dict) -> None:
    try:
        run_config = _load_json(path.read_text(), path)
    except OSError as error:
        raise ValueError(f"{path}: cannot read run_config.json") from error
    if run_config.get("mode") != "train":
        raise ValueError("run_config.json mode must be train")
    for field in (
        "baseline_id", "model_config", "delta_config", "parameter_count",
        "fp32_parameter_bytes",
    ):
        if run_config.get(field) != payload.get(field):
            raise ValueError(f"run_config.json {field} conflicts with checkpoint")
    recorded_training = run_config.get("training_config")
    if not isinstance(recorded_training, dict):
        raise ValueError("run_config.json training_config is missing")
    for field in STABLE_TRAINING_FIELDS:
        recorded = recorded_training.get(field)
        checkpoint_value = training_config[field]
        if field == "betas" and isinstance(recorded, list):
            recorded = tuple(recorded)
        if recorded != checkpoint_value:
            raise ValueError(f"run_config.json training_config {field} conflicts with checkpoint")


def validate_resume_files(directory: Path, payload: dict) -> tuple[int, dict | None, bool]:
    training_config = validate_training_config(payload)
    history_path = directory / "history.jsonl"
    try:
        lines = history_path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"{history_path}: cannot read history.jsonl") from error
    if not lines:
        raise ValueError(f"{history_path}: history.jsonl is empty")
    best_epoch = 0
    best_validation = None
    best_rmse = float("inf")
    previous_epoch = 0
    last_record = None
    for line_number, line in enumerate(lines, 1):
        record = _load_json(line, history_path)
        epoch = record.get("epoch")
        step = record.get("global_step")
        validation = record.get("validation")
        if (type(epoch) is not int or epoch != previous_epoch + 1
                or type(step) is not int or step < 0
                or not isinstance(validation, dict)
                or not isinstance(validation.get("rmse_cm"), dict)):
            raise ValueError(f"{history_path}:{line_number}: invalid epoch record")
        rmse = _finite_metric(validation["rmse_cm"].get("all"), "validation RMSE")
        if rmse < best_rmse:
            best_rmse = rmse
            best_epoch = epoch
            best_validation = validation
        recorded_best = _finite_metric(record.get("best_validation_rmse_cm"), "best RMSE")
        if recorded_best != best_rmse:
            raise ValueError(f"{history_path}:{line_number}: best RMSE conflicts with history")
        previous_epoch = epoch
        last_record = record
    assert last_record is not None
    checkpoint_best = payload.get("best_validation_rmse_cm")
    if (last_record["epoch"] != payload.get("epoch")
            or last_record["global_step"] != payload.get("global_step")
            or last_record["best_validation_rmse_cm"] != checkpoint_best):
        raise ValueError("history.jsonl last epoch, global_step, or best metric conflicts with checkpoint")
    config_path = directory / "run_config.json"
    config_present = config_path.exists()
    if config_present:
        _validate_run_config(config_path, payload, training_config)
    return best_epoch, best_validation, config_present
