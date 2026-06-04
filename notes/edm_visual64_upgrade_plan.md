# EDM and Visual64 Quality Upgrade

## Motivation

The current best CIFAR-10 DDPM/DDIM model has good quantitative metrics:

```text
FID: 13.3722
Inception Score: 5.2858 +/- 0.1446
```

However, CIFAR-10 is only 32x32, and the generated samples still look soft in a README. The next quality push has two targeted branches:

1. EDM-style CIFAR-10 training for sharper and more stable generation.
2. A 64x64 visual dataset branch for stronger visual presentation.

## Branch 1: CIFAR-10 EDM

Files:

```text
src/edm.py
labs/lab_cifar_flow/train_cifar10_edm_unet.py
labs/lab_cifar_flow/sample_cifar10_edm.py
labs/lab_cifar_flow/eval_cifar10_edm_metrics.py
configs/cifar10_unet_edm_500ep.yaml
configs/cifar10_edm_sampling_500ep.yaml
configs/cifar10_unet_edm_resume_300ep.yaml
configs/cifar10_edm_sampling_resume_300ep.yaml
configs/cifar10_metrics_edm_resume_300ep_1k.yaml
configs/cifar10_metrics_edm_resume_300ep_5k.yaml
```

Core changes:

- Log-normal sigma sampling.
- EDM preconditioned denoising:

```text
D_theta(x, sigma, y) = c_skip x + c_out F_theta(c_in x, c_noise, y)
```

- Karras sigma schedule.
- Euler / Heun EDM sampler.
- CFG through conditional/unconditional denoiser outputs.

Smoke validation completed:

```text
checkpoints/cifar10_unet_edm_smoke.pt
figures/cifar_flow/edm_cifar10_smoke_samples.png
results/cifar_flow/config_used_edm_smoke.json
results/cifar_flow/sample_config_edm_cifar10_smoke.json
```

Original main launch:

```bash
tmux new -s cifar10_edm_500ep
CUDA_VISIBLE_DEVICES=1 conda run -n fm_diffusion python labs/lab_cifar_flow/train_cifar10_edm_unet.py \
  --config configs/cifar10_unet_edm_500ep.yaml 2>&1 | tee logs/cifar10_edm_500ep.log
```

## Branch 2: Visual64 EDM

Files:

```text
src/visual_dataset.py
labs/lab_visual64/train_visual64_edm_unet.py
labs/lab_visual64/sample_visual64_edm.py
configs/visual64_pets_edm_300ep.yaml
configs/visual64_pets_edm_sampling_300ep.yaml
```

Supported data modes:

```text
dataset: pets
dataset: imagefolder
```

The ImageFolder path should have the standard structure:

```text
data/my_visual64/
├── class_a/
├── class_b/
└── class_c/
```

Smoke validation completed with a temporary ImageFolder dataset:

```text
checkpoints/visual64_imagefolder_edm_smoke.pt
figures/visual64/imagefolder_edm_smoke_samples.png
results/visual64/config_used_imagefolder_edm_smoke.json
results/visual64/sample_config_imagefolder_edm_smoke.json
```

Pets dataset initialization initially stalled while relying on the torchvision wrapper. The downloaded image archive was partially extractable and produced 7192 usable images, enough for the visual branch. These images were organized into:

```text
data/visual64/pets_imagefolder
```

This is now the preferred visual64 launch path. Use either:

- a prepared ImageFolder dataset, or
- rerun Pets once network/download availability is confirmed.

Recommended ImageFolder Pets launch:

```bash
tmux new -s visual64_pets_imagefolder_edm_300ep
CUDA_VISIBLE_DEVICES=2 conda run -n fm_diffusion python labs/lab_visual64/train_visual64_edm_unet.py \
  --config configs/visual64_pets_imagefolder_edm_300ep.yaml 2>&1 | tee logs/visual64_pets_imagefolder_edm_300ep.log
```

## Branch 3: Visual64 Binary Pets EDM

The 37-class Pets setup is useful but difficult for fast visual improvement. A simpler binary Cats vs Dogs split was created from the same extracted Oxford Pets images:

```text
data/visual64/pets_binary_imagefolder
cat: 2350 images
dog: 4839 images
```

This branch is expected to form recognizable 64x64 pet samples faster than the 37-class version.

Files:

```text
configs/visual64_pets_binary_edm_300ep.yaml
configs/visual64_pets_binary_edm_sampling_300ep.yaml
```

Recommended launch:

```bash
tmux new -s visual64_pets_binary_edm_300ep
CUDA_VISIBLE_DEVICES=3 conda run -n fm_diffusion python labs/lab_visual64/train_visual64_edm_unet.py \
  --config configs/visual64_pets_binary_edm_300ep.yaml 2>&1 | tee logs/visual64_pets_binary_edm_300ep.log
```

Sampling:

```bash
CUDA_VISIBLE_DEVICES=3 conda run -n fm_diffusion python labs/lab_visual64/sample_visual64_edm.py \
  --config configs/visual64_pets_binary_edm_sampling_300ep.yaml
```

## Final Status

CIFAR-10 EDM resume run completed:

```text
checkpoint: checkpoints/cifar10_unet_edm_resume_300ep.pt
epoch: 300
step: 155600
sample: figures/cifar_flow/edm_cifar10_resume_300ep_heun40_cfg2_upscaled.png
metric json: results/cifar_flow/metrics_edm_resume_300ep_1k.json
FID@1k: 41.4960
Inception Score@1k: 5.0062 +/- 0.4826
```

The EDM run produced more coherent CIFAR-10 class structure than the early checkpoints, but it did not beat the stronger DDPM/DDIM cosine v-prediction baseline:

```text
DDPM/DDIM FID@5k: 13.3722
DDPM/DDIM IS@5k: 5.2858 +/- 0.1446
```

Therefore:

```text
CIFAR-10 quantitative baseline: keep DDPM/DDIM cosine v-pred.
CIFAR-10 EDM: keep as completed quality-upgrade attempt and engineering evidence.
README visual showcase: use Visual64 EDM Pets.
```

Intermediate CIFAR-10 EDM samples were generated at around epoch 20:

```text
figures/cifar_flow/edm_cifar10_ep20_raw_heun40_cfg2_upscaled.png
figures/cifar_flow/edm_cifar10_ep20_ema_heun40_cfg2_upscaled.png
```

The first EDM launch was interrupted after an epoch-20 checkpoint, so a resume run was started:

```bash
tmux new -s cifar10_edm_resume_300ep
CUDA_VISIBLE_DEVICES=5 conda run -n fm_diffusion python labs/lab_cifar_flow/train_cifar10_edm_unet.py \
  --config configs/cifar10_unet_edm_resume_300ep.yaml 2>&1 | tee logs/cifar10_edm_resume_300ep.log
```

Monitor the resume run from the CSV log rather than the tmux pane, because stdout can be buffered:

```bash
tail -f results/cifar_flow/training_log_edm_resume_300ep.csv
```

Intermediate resume samples were generated at around epoch 155:

```text
figures/cifar_flow/edm_cifar10_resume_ep155_raw_heun40_cfg2_upscaled.png
figures/cifar_flow/edm_cifar10_resume_ep155_ema_heun40_cfg2_upscaled.png
```

The final 300-epoch checkpoint was evaluated with:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n fm_diffusion python labs/lab_cifar_flow/sample_cifar10_edm.py \
  --config configs/cifar10_edm_sampling_resume_300ep.yaml

CUDA_VISIBLE_DEVICES=6 conda run -n fm_diffusion python labs/lab_cifar_flow/eval_cifar10_edm_metrics.py \
  --config configs/cifar10_metrics_edm_resume_300ep_1k.yaml
```

For unattended completion, use the finalizer script:

```bash
tmux new -s cifar10_edm_finalizer
/home/ldz/miniconda3/envs/fm_diffusion/bin/python labs/lab_cifar_flow/finalize_cifar10_edm_resume.py \
  --gpu 0 \
  --target_epoch 300
```

The formal 5k metric is optional because the 1k result is already well behind the existing DDPM/DDIM FID. Run it only if a same-sample-count comparison table is required:

```bash
CUDA_VISIBLE_DEVICES=6 conda run -n fm_diffusion python labs/lab_cifar_flow/eval_cifar10_edm_metrics.py \
  --config configs/cifar10_metrics_edm_resume_300ep_5k.yaml
```

Intermediate 37-class Pets samples were generated at around epoch 29:

```text
figures/visual64/pets_imagefolder_edm_ep29_raw_heun50_cfg2.png
figures/visual64/pets_imagefolder_edm_ep29_ema_heun50_cfg2.png
```

Alternative torchvision Pets launch:

```bash
tmux new -s visual64_pets_edm_300ep
CUDA_VISIBLE_DEVICES=2 conda run -n fm_diffusion python labs/lab_visual64/train_visual64_edm_unet.py \
  --config configs/visual64_pets_edm_300ep.yaml 2>&1 | tee logs/visual64_pets_edm_300ep.log
```

## Priority

The quality-upgrade branch is complete. The highest-value project presentation choice is:

```text
1. Use DDPM/DDIM cosine v-pred for CIFAR-10 quantitative claims.
2. Use Visual64 EDM Pets for visual clarity and README screenshots.
3. Mention CIFAR-10 EDM as an attempted quality upgrade that improved visual structure but did not improve FID.
```
