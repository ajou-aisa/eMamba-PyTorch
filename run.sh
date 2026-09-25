#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate

# for train fp32
#
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --mode train \
#   --device cuda \
#   --epochs 100 \
#   --batch-size 64 \
#   --lr 0.001 \
#   --seed 0 \
#   --output-dir results/0922_emamba_100ep_seed0


# for test fp32
#
#  .venv/bin/python train.py \
#    --mode eval \
#    --checkpoint results/0922_1653_emamba_100ep_seed0/best.pt \
#    --split test --device cuda --batch-size 64 --json-stdout


# fp32 -> ptq -> quantized.pt (using calibration -> decide scale)
#
#  .venv/bin/python ptq_emamba.py \
#    --checkpoint results/0922_1653_emamba_100ep_seed0/best.pt \
#    --output-dir results/0922_1653_emamba_100ep_seed0_ptq_run02 \
#    --split validation \
#    --device cuda \
#    --batch-size 64


# for test quantized model
# PTQ QEMamba:
#
#  .venv/bin/python ptq_emamba.py \
#    --artifact results/0922_1653_emamba_100ep_seed0_ptq_run02/quantized.pt \
#    --split test --device cuda --batch-size 64
