#!/usr/bin/env bash
set -euo pipefail

python labs/lab5/04_sample_discrete_denoiser.py \
  --ckpt checkpoints/discrete_denoiser_toy.pt \
  --device cpu
