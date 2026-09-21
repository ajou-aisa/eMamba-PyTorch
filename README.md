# eMamba PyTorch: provisional FP32 MARS baseline

`provisional_fp32_v1` is a trainable FP32 baseline for public MARS data.
It is **not** a verified reconstruction of the authors' model. Details absent
from the paper remain reproduction choices pending author confirmation.

## Structure

```text
datasets/mars.py              MARS NumPy loader and integrity checks
models/patch_embedding.py     row-major 2×2 flattening
models/mamba/block.py         RangeNorm, gate, causal depthwise Conv, SSM, residual
models/mamba/range_norm.py    range normalization over D
models/mamba/selective_ssm.py sequential selective recurrence
models/emamba.py              Patch, two blocks, OutputHead
models/output_head.py         mean or last readout and one Linear
train.py                      CLI, device/data setup, smoke/train/eval control
training/engine.py            PyTorch train epoch and inference evaluation
training/metrics.py           MARS CPU FP64 tensor accumulator
training/checkpoint.py        checkpoint save/load and run metadata
training/diagnostics.py       smoke-only delta statistics
training/reporting.py         human-readable terminal summaries
tests/                        unit and pipeline checks
test_conv.py                  original causal Conv check
third_party/MARS/feature/     public MARS NumPy files from submodule
```

Paper-reported MARS dimensions: D=20, E=2 (ED=40), P=2, M=2, N=8,
and 57 outputs. Default model has **9,077 parameters** and **36,308 bytes**
of FP32 parameter data (35.46 KiB). The paper reports 67.3 KB for its
model; no layers were added to match that size. Serialized `.pt` files
also include optimizer state and metadata, so file size differs.

## Setup

```bash
git clone --recurse-submodules https://github.com/ajou-aisa/eMamba-PyTorch
cd eMamba-PyTorch
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For an existing clone, run `git submodule update --init --recursive`.
Default data path is `third_party/MARS/feature/`, resolved relative to
`train.py`. Use `--data-root /path/to/feature` for another location.
The unchanged public files are:

| Split | Feature file | Label file | Samples |
| --- | --- | --- | ---: |
| train | `featuremap_train.npy` | `labels_train.npy` | 24,066 |
| validation | `featuremap_validate.npy` | `labels_validate.npy` | 8,033 |
| test | `featuremap_test.npy` | `labels_test.npy` | 7,984 |

Features must be `[S,8,8,5]`, labels `[S,57]`. Invalid shapes, counts,
or NaN/Inf stop with a split/path error. Both arrays become FP32; labels
stay in metres during optimization.

## Run

```bash
python -m unittest discover -s tests -v
python train.py --mode smoke --steps 25 --batch-size 32 \
  --device cpu --debug-numerics
python train.py --mode train --epochs 1 --batch-size 128 \
  --device auto --output-dir results/provisional_fp32_v1
python train.py --mode eval \
  --checkpoint results/provisional_fp32_v1/best.pt \
  --split validation --device auto
# Run test evaluation only after validation checkpoint selection.
python train.py --mode eval \
  --checkpoint results/provisional_fp32_v1/best.pt \
  --split test --device auto
```

Terminal output defaults to a compact run summary, one row per epoch,
and a final best-checkpoint summary. Train and validation batch bars use
`tqdm` only when stderr is a terminal. `--no-progress` disables them;
`--json-stdout` restores the previous machine-readable JSON lines and
also disables bars. The latter works for smoke, train, and eval.

Each smoke/train output directory must be new, preventing accidental
overwrite. Smoke repeats one fixed real train batch and saves `smoke.json`
with initial/final loss and per-block delta diagnostics under separate
`results/provisional_fp32_v1_smoke/` by default. It does not evaluate
validation or test. Train shuffles all train samples, evaluates all
validation samples without shuffling, and never drops the final batch.
It writes `run_config.json` with the full start configuration,
`history.jsonl` with unchanged per-epoch raw metrics, `best.pt` selected
by validation mean RMSE,
and `last.pt` from the last completed epoch. Eval restores architecture
and readout from checkpoint. Checkpoints load on CPU before the model moves
to the requested device. Training resume is not implemented.

Training uses a standard PyTorch loop with a caller-owned MSE criterion,
backward, finite-norm gradient clipping, and optimizer step. Shape, loss,
gradient norm, and final validation metric checks always run.
`--debug-numerics` additionally checks predictions, individual gradients,
CPU transfer, FP64 conversion, errors, and metric accumulators. Smoke enables
these detailed checks by default.

Defaults: one epoch, Adam (lr 0.001, betas 0.9/0.999, zero weight decay),
MSE loss, batch size 128, gradient norm cap 1.0, seed 0, `num_workers=0`,
mean readout, FP32. These are provisional training choices, not confirmed
paper settings. `--device auto` chooses CUDA, then MPS, then CPU.
Unavailable requested devices fail. On CUDA, TF32 is disabled for matmul
and convolution and the applied API/settings are printed. No AMP,
quantization, scheduler, or test-based checkpoint selection is used.

## Reproduction choices to confirm

- Patch embedding is row-major, flatten-only, with no learned projection.
- Each block has a kernel-4 depthwise Conv1d with bias. It selects the
  causal prefix after padding, has no separate post-Conv SiLU, and uses
  SiLU on the gate.
- RangeNorm normalizes over D using the current `clamp_min(eps)` formula.
  SSM keeps the joined parameter projection, dt_rank=ceil(D/16),
  `A=-exp(a_log)`, the same delta for `A_bar` and `B_bar`, sequential
  recurrence, and fresh zero state per forward.
- OutputHead defaults to mean pooling plus one `Linear(D,57)`.
  `--readout last` supports a separately trained comparison.
- **Delta activation:** `post_projection_relu` projects raw low-rank
  features, then applies ReLU. Earlier code applied ReLU before projection,
  allowing negative final delta. Nonnegative delta and finite
  `A=-exp(a_log)` give `A_bar=exp(delta*A)` in [0,1]. This does not
  guarantee overall training stability. Author placement is unconfirmed.
- **Delta initialization:** weights use uniform
  `[-0.001/sqrt(dt_rank), +0.001/sqrt(dt_rank)]`; bias uses
  `exp(Uniform(log(0.001), log(0.1)))`. These are initialization
  settings, not a clamp on actual delta, and neither confirmed paper
  values nor MARS-validated optima.

Checkpoint metadata records baseline ID, architecture/readout, delta
settings, training settings, parameter data size, Python/PyTorch/NumPy
versions, device, Git SHA/dirty status, and MARS submodule SHA when
available. Loading rejects conflicting baseline, readout, or delta
metadata. Old pre-projection-ReLU checkpoints are not this baseline.

## Metrics and verification

MARS X coordinates occupy indices 0:19, Y 19:38, Z 38:57. Evaluation
moves MPS/CUDA predictions to CPU in FP32 first, then converts to CPU FP64;
the two operations must remain separate. CPU torch.float64 tensors
accumulate absolute and squared errors for each of 57 coordinates over
the **whole** split. It computes each coordinate's MAE and RMSE, averages
19 coordinates per axis and all 57 overall, then converts metres to
centimetres. It does not take the square root of one global MSE or average
batch RMSE values.

Paper references: mean MAE **5.66 cm**, mean RMSE **7.85 cm**. On
2026-09-21, a local CPU run passed unit tests, real-MARS 25-step smoke
(loss 1.4865 to 1.0603, finite), and one full train/validation epoch.
That epoch yielded validation mean MAE **12.3355 cm** and mean RMSE
**16.2922 cm** over 8,033 samples; checkpoint reload gave the same
validation metrics. A one-step MPS smoke also completed with finite loss.
These values show the pipeline runs, not paper fidelity. A 150-epoch run,
test evaluation, CUDA execution, and author detail confirmation remain
unverified.
