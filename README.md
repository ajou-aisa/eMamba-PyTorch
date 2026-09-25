# eMamba PyTorch: provisional FP32 MARS baseline

`provisional_fp32_v2_flatten` is a trainable FP32 baseline for public MARS data.
It is **not** a verified reconstruction of the authors' model. Details absent
from the paper remain reproduction choices pending author confirmation.

## Structure

```text
datasets/mars.py              MARS NumPy loader and integrity checks
models/patch_embedding.py     row-major 2×2 flattening and Linear projection
models/mamba/block.py         RangeNorm, gate, causal depthwise Conv, SSM, residual
models/mamba/range_norm.py    range normalization over D
models/mamba/selective_ssm.py sequential selective recurrence
models/emamba.py              Patch, two blocks, OutputHead
models/piecewise.py           FP32 piecewise SiLU and exp
models/q_emamba.py            INT8 PTQ Patch, quantized blocks, OutputHead
models/output_head.py         flatten readout and a 320 -> 20 -> 57 MLP
train.py                      CLI argument parsing and entry point
training/data.py              deterministic MARS split and DataLoader setup
training/runtime.py           device, FP32 precision, and seed setup
training/session.py           fresh and resumed run initialization
training/workflow.py          smoke/train/eval orchestration
training/engine.py            PyTorch train epoch and inference evaluation
training/metrics.py           MARS CPU FP64 tensor accumulator
training/checkpoint.py        checkpoint save/load and run metadata
training/resume.py            resume configuration and history validation
training/diagnostics.py       smoke-only delta statistics
training/reporting.py         human-readable terminal summaries
tests/                        unit and pipeline checks
tests/test_conv.py            causal Conv checks
run.sh                        environment activation and commented training example
third_party/MARS/feature/     public MARS NumPy files from submodule
```

Paper-reported MARS dimensions: D=20, E=2 (ED=40), P=2, M=2, N=8,
and 57 outputs. Default model has **15,717 parameters** and **62,868 bytes**
of FP32 parameter data (61.39 KiB). The paper reports 67.3 KB for its
model; the current MLP head is an experimental choice, not a confirmed
author architecture. Serialized `.pt` files
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

The loader keeps all 7,984 official test samples in their original order.
Only the official train and validation arrays are concatenated (train first)
and randomly repartitioned 80:20 with a dedicated PyTorch generator seeded
with 0: 25,679 training and 6,420 validation samples. The overall split is
approximately 64:16:20. Features and labels use the same sample indices;
the original `.npy` files are not modified. The split seed is fixed separately
from the training `--seed`.
All six source arrays are loaded and checked even when only one split is
requested for evaluation.

Checkpoints trained with the previous all-data resplit may have seen official
test samples during training. Start a fresh run in a new output directory;
do not resume those checkpoints for an independent official-test evaluation.
Their recorded validation metrics also refer to the previous split.

Features must be `[S,8,8,5]`, labels `[S,57]`. Invalid shapes, counts,
or NaN/Inf stop with a split/path error. Both arrays become FP32; labels
stay in metres during optimization.

## Run

```bash
python -m unittest discover -s tests -v
python train.py --mode smoke --steps 25 --batch-size 32 \
  --device cpu --debug-numerics
python train.py --mode train --epochs 1 --batch-size 128 \
  --device auto --output-dir results/provisional_fp32_v2_flatten
# Longer run; use a new output directory for each experiment.
python train.py --mode train --epochs 100 --batch-size 64 \
  --lr 0.001 --seed 0 --device cuda \
  --output-dir results/emamba_v2_100ep_new
python train.py --mode eval \
  --checkpoint results/provisional_fp32_v2_flatten/best.pt \
  --split validation --device auto
# Run test evaluation only after validation checkpoint selection.
python train.py --mode eval \
  --checkpoint results/provisional_fp32_v2_flatten/best.pt \
  --split test --device auto
```

`run.sh` currently activates `.venv`; its training command is commented out.
Run the CLI directly or uncomment and adjust that example to start training.

Terminal output defaults to a compact run summary, one row per epoch,
and a final best-checkpoint summary. Train and validation batch bars use
`tqdm` only when stderr is a terminal. `--no-progress` disables them;
`--json-stdout` restores the previous machine-readable JSON lines and
also disables bars. The latter works for smoke, train, and eval.

Each smoke/train output directory must be new, preventing accidental
overwrite. Smoke repeats one fixed real train batch and saves `smoke.json`
with initial/final loss and per-block delta diagnostics under separate
`results/provisional_fp32_v2_flatten_smoke/` by default. It does not evaluate
validation or test. Train shuffles all train samples, evaluates all
validation samples without shuffling, and never drops the final batch.
It writes `run_config.json` with the full start configuration,
`history.jsonl` with unchanged per-epoch raw metrics, `best.pt` selected
by validation mean RMSE,
and `last.pt` from the last completed epoch. Eval restores architecture
and readout from checkpoint. Checkpoints load on CPU before the model moves
to the requested device.
Eval defaults to the validation split, batch size 128, and zero workers;
`--batch-size` and `--num-workers` can be set explicitly for evaluation.
Smoke writes `run_config.json` and `smoke.json`, but no weight checkpoint.

## Resume training

```bash
python train.py --mode train \
  --resume results/provisional_fp32_v2_flatten/last.pt \
  --epochs 150 --device cuda
```

`--epochs` is the final target epoch: a checkpoint at epoch 20 can run epochs
21 through 150, unless early stopping ends the run sooner. Resume accepts
only `last.pt`, reuses its parent output
directory, and appends to `history.jsonl`. An explicit `--output-dir` must
name that same directory. Model, optimizer state (AdamW or legacy Adam),
global step, saved best RMSE,
batch size, learning rate, seed, and other training settings come from the
checkpoint; CLI hyperparameter overrides are rejected. The existing
`run_config.json` is checked and preserved. Device changes are allowed,
but FP32 remains required. Resume starts at the next **whole epoch**;
mid-epoch batches are not restored.

New `training_config` records a SHA-256 fingerprint of the four train and
validation feature/label `.npy` files. Resume rejects changed training data;
older checkpoints without this fingerprint warn that data identity is
unverified. Resume also checks `best.pt` against `last.pt` and the history,
including its epoch, best score, nonlinear policy, and training settings.

New checkpoints include Python, NumPy, Torch, and available CUDA/MPS RNG
state. The current train DataLoader uses Torch's global RNG for shuffling,
so restoring that state preserves its order. Older checkpoints without
RNG state remain usable for evaluation and may resume with a prominent
warning; their continuation cannot be bitwise identical to an uninterrupted
run. Cross-device continuation is supported, though backend arithmetic and
unavailable target-device RNG state can also prevent bitwise equivalence.

New checkpoints save scheduler state. Resuming with the same planned
`--epochs` restores its position exactly. Changing the epoch target, or
resuming an older checkpoint without scheduler state, restarts the cosine
schedule from the saved optimizer learning rate with a warning; that path is
not bitwise equivalent to an uninterrupted run. Early stopping uses the best
epoch recovered from `history.jsonl`.

## Training behavior

Training uses a standard PyTorch loop with a caller-owned MSE criterion,
backward, finite-norm gradient clipping, and optimizer step. Shape, loss,
gradient norm, and final validation metric checks always run.
`--debug-numerics` additionally checks predictions, individual gradients,
CPU transfer, FP64 conversion, errors, and metric accumulators. Smoke enables
these detailed checks by default.

Defaults: one epoch, AdamW (lr 0.001, betas 0.9/0.999, weight decay 0.01),
MSE loss, batch size 128, gradient norm cap 1.0, seed 0, `num_workers=0`,
flatten readout, FP32 with piecewise SiLU/exp. These are provisional training
choices, not confirmed paper settings. `--device auto` chooses CUDA, then
MPS, then CPU.
Unavailable requested devices fail. On CUDA, TF32 is disabled for matmul
and convolution and the applied API/settings are printed. No AMP,
quantization, QAT, or test-based checkpoint selection is implemented.

Train mode uses `CosineAnnealingLR(T_max=epochs, eta_min=1e-6)`, stepped
once after each epoch's validation and before saving checkpoints. The
optimizer learning rate in a checkpoint is therefore the rate for the next
epoch; `training_config.lr` records the initial rate. Smoke does not use a
scheduler.

Training stops after **15 consecutive epochs without a strictly lower
validation mean RMSE**. The stopping epoch is still saved to `last.pt` and
`history.jsonl`; `best.pt` remains the checkpoint with the lowest validation
mean RMSE. Scheduler and early-stopping settings are fixed in `training/workflow.py`,
without dedicated CLI options.

## Reproduction choices to confirm

- Patch embedding flattens each 2×2×5 patch in row-major order, then applies
  a shared `Linear(20,20)` with bias to all 16 tokens, matching sway.
  This adds 420 parameters and preserves the `[B,16,20]` output shape.
- Each block has a kernel-4 depthwise Conv1d with bias. It selects the
  causal prefix after padding, has no separate post-Conv SiLU, and uses
  SiLU on the gate. Input, gate, and output Linear projections have no bias;
  convolution and delta projection retain their biases.
- RangeNorm normalizes over D using the current `clamp_min(eps)` formula.
  SSM keeps the joined parameter projection, dt_rank=ceil(D/16),
  `A=-exp(a_log)`, the same delta for `A_bar` and `B_bar`, sequential
  recurrence, and fresh zero state per forward.
- OutputHead uses the sway-style
  `[B,16,20] -> Flatten(320) -> Linear(320,20) -> ReLU -> Linear(20,57)`,
  with bias in both Linear layers. Flatten preserves token-major order
  and keeps the batch dimension. The head requires exactly 16 tokens
  for the default MARS configuration; mean/last readout is not supported.
  The backbone, including patch embedding, has 8,100 parameters and the head
  has 7,617 parameters. Checkpoints without the learned patch projection or
  with block projection biases are incompatible and are rejected by the
  parameter-count check; start a new training run.
  Earlier pooled MLP and single-Linear checkpoints are incompatible;
  start a new training run in a new output directory. The new baseline ID
  rejects old checkpoints before loading their weights.
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

## Checkpoint contents

Both `best.pt` and `last.pt` contain a dictionary with the following fields
for current training runs:

| Key | Contents |
| --- | --- |
| `model_state_dict` | 30 FP32 parameter tensors, totaling 15,717 values |
| `optimizer_state_dict` | Per-parameter `step`, `exp_avg`, `exp_avg_sq`, plus optimizer parameter groups |
| `epoch` | Last completed epoch represented by this checkpoint |
| `global_step` | Number of optimizer steps completed |
| `best_validation_rmse_cm` | Lowest validation mean RMSE observed so far, in cm |
| `baseline_id` | `provisional_fp32_v2_flatten` |
| `nonlinear_policy` | `piecewise_fp32` for fresh runs; saved policy is restored on load |
| `model_config` | `d_model`, `expand`, `patch_size`, `num_blocks`, `d_state`, `out_dim`, `in_channels`, `readout` |
| `readout` | `flatten` |
| `training_config` | Precision, optimizer, initial learning rate, betas, weight decay, loss, batch size, gradient clip, seed, workers, epochs, steps, TF32 settings, train/validation data SHA-256 |
| `delta_config` | Delta activation and initialization settings |
| `parameter_count` | 15,717 |
| `fp32_parameter_bytes` | 62,868; parameter storage only, not serialized file size |
| `environment` | Python, PyTorch, NumPy versions and training device |
| `git` | Commit, dirty flag, and MARS submodule commit when available |
| `rng_state` | Python, NumPy, Torch, available CUDA/MPS states, and device RNG support flags |
| `scheduler_state_dict` | Cosine scheduler state, including its epoch position and planned target |
| `scheduler_epoch_offset` | Global epoch offset when a resumed run restarted its cosine schedule |

The model tensors include Linear/Conv weights and biases, RangeNorm
`gamma`/`beta`, and SSM `a_log`/`d_skip`. The effective `A=-exp(a_log)` is
computed during forward; input-dependent B, C, delta, activations, and
recurrent hidden states are not stored as model parameters. INT8 weights and
quantization scales/zero-points are not included.

Loading uses `torch.load(..., map_location="cpu", weights_only=True)` and
checks baseline, model/readout, delta settings, parameter count/size, and
the model state dictionary. Older v1, pooled-readout, or
pre-projection-ReLU checkpoints are incompatible with the current model.
Older compatible checkpoints without `nonlinear_policy` are inferred only
when clean, known Git provenance identifies their policy. If provenance is
ambiguous, pass `--legacy-nonlinear native_fp32` or
`--legacy-nonlinear piecewise_fp32` to `train.py` evaluation/resume or
`ptq_emamba.py` conversion, according to the checkpoint's original behavior.

## Metrics and verification

MARS X coordinates occupy indices 0:19, Y 19:38, Z 38:57. Evaluation
moves MPS/CUDA predictions to CPU in FP32 first, then converts to CPU FP64;
the two operations must remain separate. CPU torch.float64 tensors
accumulate absolute and squared errors for each of 57 coordinates over
the **whole** split. It computes each coordinate's MAE and RMSE, averages
19 coordinates per axis and all 57 overall, then converts metres to
centimetres. It does not take the square root of one global MSE or average
batch RMSE values.

Paper references: mean MAE **5.66 cm**, mean RMSE **7.85 cm**. Local
validation results below are not a like-for-like paper/test comparison.

The following files were present locally on 2026-09-23; they are not
guaranteed to be included in a fresh clone. Values come from each run's
`best.pt` and `history.jsonl`, not a new evaluation:

| Run under `results/` | Baseline | Best epoch | Validation samples | Mean MAE (cm) | Mean RMSE (cm) |
| --- | --- | ---: | ---: | ---: | ---: |
| `0922_1653_emamba_100ep_seed0` | v2 flatten | 100 | 6,420 | 5.7771 | 7.9362 |
| `0922_emamba_100ep_seed0` | v2 flatten | 95 | 6,414 | 5.8234 | 7.9927 |
| `emamba_100ep_seed0` | v1 mean | 99 | 6,414 | 6.5630 | 8.8818 |

Each directory contains `best.pt`, `last.pt`, `run_config.json`, and
`history.jsonl`. All three histories record 100 completed epochs, and
their checkpoints record CUDA, AdamW, batch size 64, and seed 0.
The v1 checkpoint has 9,077 parameters and cannot load into the current
model. The two runs with 6,414 validation samples used an earlier split;
their scores should not be compared directly with the current split or
treated as evidence of an untouched official test set. Checkpoint metadata
does not store split indices or a split-policy version.

For a validation evaluation of the local v2 checkpoint matching the current
split size:

```bash
python train.py --mode eval \
  --checkpoint results/0922_1653_emamba_100ep_seed0/best.pt \
  --split validation --batch-size 64 --device auto
```

These stored results establish completed local training, not paper fidelity.
They do not establish official-test performance or INT8/QAT accuracy.

## INT8 post-training quantization

Install the pinned PTQ additions into the existing environment, then convert a
checkpoint into a new result directory:

```bash
.venv/bin/python -m pip install -r requirements-ptq.txt
.venv/bin/python -m pytest -q tests test/ptq
.venv/bin/python ptq_emamba.py \
  --checkpoint results/0922_1653_emamba_100ep_seed0/best.pt \
  --output-dir results/0922_1653_emamba_100ep_seed0_ptq_run01
```

The default split is `validation`; add `--split test` only for an explicitly
requested final test report. Calibration always uses the deterministic 2,048
frame seed-0 training subset, while max-versus-99.9-percentile profile selection
always uses the full validation split. `--device auto` prefers CUDA and falls
back to CPU. The destination must not exist, preventing result overwrite.

Conversion writes only `quantized.pt` and `metrics.json`. The checkpoint stores
contiguous INT8 parameter codes, power-of-two scales, model configuration,
calibration and dataset fingerprints, numeric policy, and version metadata. A
source-free reload reconstructs `models.q_emamba.QEMamba`. It reuses the FP32
`PatchEmbedding`, `OutputHead`, and causal `MambaConv1D` implementations while
replacing their internal Linear/Conv layers and Mamba blocks with PTQ modules.
The frozen model can be evaluated with:

```bash
.venv/bin/python ptq_emamba.py \
  --artifact results/0922_1653_emamba_100ep_seed0_ptq_run01/quantized.pt \
  --split validation
```

Metrics are computed from dequantized coordinates in centimetres through the
same evaluator as the FP32 baseline. Frozen Linear and depthwise Conv use INT8
operand codes with INT64 multiplication and accumulation. Bias codes are aligned
to a common scale before integer addition; outputs are rounded ties-to-even,
clipped to INT8, and dequantized to FP32 for the next layer. Calibration retains
the floating-point path. RangeNorm and SSM state arithmetic remain integer;
SiLU, exponential, delta products, gating, and residual addition remain FP32
with quantized boundaries; SiLU/exp use the selected native or piecewise
policy. It is not a claim of fully
integer hardware execution or guaranteed reproduction of paper accuracy.

RangeNorm follows the original operation order: mean, centering, centered
range with epsilon, normalization division, gamma multiplication, then beta
addition. Mean/centered values use the input-code scale with 24 fractional bits;
normalized values use scale `2^-24`, all held in INT64 with ties-to-even
rounding. Only the final output is clipped to INT8 and dequantized to FP32.
This replaces the earlier combined-numerator implementation; fixed-point
rounding need not be bit-identical to FP32 or to that earlier implementation.

The following historical results used FP32 Linear/Conv arithmetic, before the
INT64 affine and sequential RangeNorm changes. Re-evaluate existing artifacts
to measure the current code; their INT8 parameters and scales remain compatible.

The local `run01` selected the percentile profile on all 6,420 validation
samples. Its saved profile hash is
`3cfaf2a2fa1a7daea7330ec69902de144956cc5cc03664bcff25a6493596f6f9`, and
source-free reload reproduced output codes on every validation frame.

| Split | Model | Samples | Mean MAE (cm) | Mean RMSE (cm) |
| --- | --- | ---: | ---: | ---: |
| validation | FP32 | 6,420 | 5.7771 | 7.9362 |
| validation | PTQ | 6,420 | 6.1072 | 8.2221 |
| test | FP32 | 7,984 | 5.9167 | 8.1012 |
| test | frozen PTQ artifact | 7,984 | 6.2414 | 8.3873 |

The test row is an explicitly requested frozen-artifact evaluation; test data
did not participate in calibration or profile selection.

### Piecewise SiLU and exponential

PTQ conversion defaults to the source checkpoint's nonlinear policy. Fresh
FP32 checkpoints use the sway software model's 17-segment SiLU and 11-segment
exponential approximation, so no flag is needed for them. Use `--piecewise`
to override a native checkpoint, or `--native` to override a piecewise one:

```bash
.venv/bin/python ptq_emamba.py \
  --checkpoint results/0922_1653_emamba_100ep_seed0/best.pt \
  --output-dir results/0922_1653_emamba_100ep_seed0_ptq_pwl_run01 \
  --piecewise --split validation --device cuda --batch-size 64
```

Use a fresh output directory for each conversion. The approximation applies to
the existing gate SiLU and `Abar = exp(delta * A)` in both calibration and frozen
inference. Delta retains ReLU; continuous `A = -exp(a_log)` is unchanged.
The functions interpolate in FP32, then use the existing INT8 boundaries, without
an input-code lookup table. SiLU uses 18 knots over `[-7,7]`, with zero/identity
tails; exp uses 12 knots over `[-4,1]`, with zero/e tails. These are sway's own
secant coefficients, not published author coefficients.

The artifact records its nonlinear mode. Piecewise artifacts also record knots,
interpolation version, and tail rules. Evaluate with
`--artifact <path>/quantized.pt --split test --device cuda`;
the saved mode is restored automatically, and nonlinear overrides are not
accepted on reload.
