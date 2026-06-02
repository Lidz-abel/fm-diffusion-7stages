#!/usr/bin/env bash
set -euo pipefail

python labs/lab5/01_run_nfe_compare.py \
  --ckpt checkpoints/stage2_flow_matching.pt \
  --nfe 5 10 20 50 100 \
  --num_samples 4096 \
  --device cpu
