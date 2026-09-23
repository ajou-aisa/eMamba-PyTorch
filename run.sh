#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source .venv/bin/activate

# for train fp32
# CUDA_VISIBLE_DEVICES=0 python train.py \
#   --mode train \
#   --device cuda \
#   --epochs 100 \
#   --batch-size 64 \
#   --lr 0.001 \
#   --seed 0 \
#   --output-dir results/0922_emamba_100ep_seed0
