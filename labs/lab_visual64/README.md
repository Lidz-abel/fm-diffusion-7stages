# Visual64 EDM Showcase Lab

This lab is the visual-quality extension branch. It is separate from CIFAR-10 metrics and is meant to produce more readable 64x64 showcase samples.

## Goal

Train a class-conditional EDM-style U-Net on a 64x64 visual dataset.

The default dataset is Oxford-IIIT Pets:

```text
dataset: pets
image_size: 64
model: Attention U-Net
training objective: EDM preconditioned denoising
sampler: Karras sigma schedule + Heun
```

## Smoke Test

Run a short test before any long training:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n fm_diffusion python labs/lab_visual64/train_visual64_edm_unet.py \
  --config configs/visual64_pets_edm_smoke.yaml \
  --max_steps 5
```

Expected outputs:

```text
checkpoints/visual64_pets_unet_edm_smoke.pt
results/visual64/config_used_pets_edm_smoke.json
results/visual64/training_log_pets_edm_smoke.csv
figures/visual64/training_curve_pets_edm_smoke.png
```

## Main Training

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n fm_diffusion python labs/lab_visual64/train_visual64_edm_unet.py \
  --config configs/visual64_pets_edm_300ep.yaml
```

Recommended tmux launch:

```bash
tmux new -s visual64_pets_edm_300ep
CUDA_VISIBLE_DEVICES=2 conda run -n fm_diffusion python labs/lab_visual64/train_visual64_edm_unet.py \
  --config configs/visual64_pets_edm_300ep.yaml 2>&1 | tee logs/visual64_pets_edm_300ep.log
```

## Sampling

```bash
CUDA_VISIBLE_DEVICES=2 conda run -n fm_diffusion python labs/lab_visual64/sample_visual64_edm.py \
  --config configs/visual64_pets_edm_sampling_300ep.yaml
```

Expected output:

```text
figures/visual64/pets64_edm_300ep_heun50_cfg2.png
results/visual64/sample_config_pets64_edm_300ep_heun50_cfg2.json
```

## Why This Branch Exists

CIFAR-10 is useful for FID and Inception Score, but 32x32 images remain visually soft when shown in a README. This 64x64 branch is for stronger visual presentation. It should not replace CIFAR-10 metrics; it complements them.
