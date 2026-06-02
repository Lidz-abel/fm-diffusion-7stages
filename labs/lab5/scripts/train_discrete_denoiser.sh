#!/usr/bin/env bash
set -euo pipefail

python labs/lab5/03_train_discrete_denoiser.py \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-3 \
  --device cpu
