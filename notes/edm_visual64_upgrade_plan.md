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
configs/cifar10_unet_edm_500ep.yaml
configs/cifar10_edm_sampling_500ep.yaml
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

Recommended main launch:

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

Alternative torchvision Pets launch:

```bash
tmux new -s visual64_pets_edm_300ep
CUDA_VISIBLE_DEVICES=2 conda run -n fm_diffusion python labs/lab_visual64/train_visual64_edm_unet.py \
  --config configs/visual64_pets_edm_300ep.yaml 2>&1 | tee logs/visual64_pets_edm_300ep.log
```

## Priority

The highest-value next run is CIFAR-10 EDM 500ep, because it directly targets the current softness while preserving metric comparability. Visual64 is the second branch for README visual impact after a real 64x64 dataset is available.
