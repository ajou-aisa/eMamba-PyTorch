from pathlib import Path
from typing import Final


RULE: Final = "─" * 60


def print_run_summary(
    config: dict, output_dir: Path, sample_counts: dict[str, int],
) -> None:
    model = config["model_config"]
    delta = config["delta_config"]
    training = config["training_config"]
    activation = delta["delta_activation"]
    delta_label = "Linear → ReLU" if activation == "post_projection_relu" else activation
    model_label = (
        f"D={model['d_model']}  E={model['expand']}  "
        f"M={model['num_blocks']}  N={model['d_state']}  readout={model['readout']}"
    )
    dataset_label = "  ".join(
        f"{split}={count:,}" for split, count in sample_counts.items()
    )
    print(f"eMamba FP32 {'Training' if config['mode'] == 'train' else 'Smoke Test'}")
    print(RULE)
    print(f"{'Baseline':<13}{config['baseline_id']}")
    print(f"{'Device':<13}{config['device']}")
    print(f"{'Model':<13}{model_label}")
    print(f"{'Parameters':<13}{config['parameter_count']:,}  "
          f"({config['fp32_parameter_bytes'] / 1024:.2f} KiB FP32)")
    print(f"{'Delta':<13}{delta_label}")
    print(f"{'Delta init':<13}[{delta['dt_init_min']}, {delta['dt_init_max']}]")
    print(f"{'Optimizer':<13}{training['optimizer']}  lr={training['lr']}  "
          f"batch={training['batch_size']}")
    duration = "Epochs" if config["mode"] == "train" else "Steps"
    print(f"{duration:<13}{training[duration.lower()]}")
    print(f"{'Seed':<13}{training['seed']}")
    print(f"{'Dataset':<13}{dataset_label}")
    print(f"{'Output':<13}{output_dir}")
    print(RULE)


def print_epoch_header() -> None:
    print("Epoch    Train Loss    Val MAE (cm)    Val RMSE (cm)    Best RMSE (cm)")
    print(RULE)


def print_epoch_result(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    validation: dict,
    best_rmse_cm: float,
    is_best: bool,
    *,
    show_axis_metrics: bool = False,
) -> None:
    mae = validation["mae_cm"]["all"]
    rmse = validation["rmse_cm"]["all"]
    print(
        f"{epoch:>2}/{total_epochs:<2}    {train_loss:>10.6f}    "
        f"{mae:>12.3f}    {rmse:>13.3f}    {best_rmse_cm:>14.3f}"
        f"  {'*' if is_best else ''}"
    )
    if show_axis_metrics:
        axes = validation["rmse_cm"]
        print(f"           RMSE  X={axes['x']:.3f}  Y={axes['y']:.3f}  Z={axes['z']:.3f}")


def print_training_summary(
    best_epoch: int, best_validation: dict, last_validation: dict, checkpoint: Path,
) -> None:
    print(RULE)
    print("Training complete")
    print(f"{'Best epoch':<16}{best_epoch}")
    print(f"{'Best MAE':<16}{best_validation['mae_cm']['all']:.3f} cm")
    print(f"{'Best RMSE':<16}{best_validation['rmse_cm']['all']:.3f} cm")
    print(f"{'Last RMSE':<16}{last_validation['rmse_cm']['all']:.3f} cm")
    print(f"{'Checkpoint':<16}{checkpoint}")
    print(RULE)


def print_resume_summary(
    checkpoint: Path, payload: dict, target_epoch: int, device: str,
) -> None:
    completed = payload["epoch"]
    print("eMamba FP32 Training — Resume")
    print(RULE)
    print(f"{'Checkpoint':<17}{checkpoint}")
    print(f"{'Completed epoch':<17}{completed}")
    print(f"{'Next epoch':<17}{completed + 1}")
    print(f"{'Target epoch':<17}{target_epoch}")
    print(f"{'Global step':<17}{payload['global_step']:,}")
    print(f"{'Best RMSE':<17}{payload['best_validation_rmse_cm']:.3f} cm")
    print(f"{'Device':<17}{device} (from {payload['environment']['device']})")
    print(RULE)


def print_smoke_summary(config: dict, result: dict) -> None:
    print("eMamba Smoke Test")
    print(RULE)
    print(f"{'Device':<14}{config['device']}")
    print(f"{'Steps':<14}{result['steps']}")
    print(f"{'Batch':<14}{config['training_config']['batch_size']}")
    print(f"{'Initial loss':<14}{result['initial_loss']:.6f}")
    print(f"{'Final loss':<14}{result['final_loss']:.6f}")
    for index, (initial, final) in enumerate(
        zip(result["delta_initial"], result["delta_final"], strict=True)
    ):
        print(f"\nDelta Block {index}")
        print(f"  Initial     mean={initial['mean']:.5f}  "
              f"zero={initial['zero_fraction']:.1%}")
        print(f"  Final       mean={final['mean']:.5f}  "
              f"zero={final['zero_fraction']:.1%}")
    print(f"\n{'Status':<14}{'PASS' if result['finite'] else 'FAIL'}")
    print(RULE)


def print_eval_summary(result: dict, device: str) -> None:
    print("eMamba Evaluation")
    print(RULE)
    print(f"{'Checkpoint':<14}{result['checkpoint']}")
    print(f"{'Device':<14}{device}")
    print(f"{'Split':<14}{result['split']}")
    print(f"{'Samples':<14}{result['samples']:,}")
    print()
    print(f"{'':<12}{'MAE (cm)':>12}    {'RMSE (cm)':>12}")
    for label, key in (("X", "x"), ("Y", "y"), ("Z", "z")):
        print(f"{label:<12}{result['mae_cm'][key]:>12.3f}    "
              f"{result['rmse_cm'][key]:>12.3f}")
    print(RULE)
    print(f"{'All':<12}{result['mae_cm']['all']:>12.3f}    "
          f"{result['rmse_cm']['all']:>12.3f}")
    print(RULE)
