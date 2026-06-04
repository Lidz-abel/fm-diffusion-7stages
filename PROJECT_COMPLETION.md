# Project Completion Note

## Current Status

This repository has reached a first complete showcase state for the diffusion / flow-matching learning project and is now in the quality-upgrade phase.

The original 7-stage learning scaffold is complete enough for study notes, toy labs, MNIST CFG, Mini-DiT, CIFAR-10 Rectified Flow, DDPM/DDIM, EDM, and Visual64 experiments.

## Best Completed Result So Far

The strongest quantitative CIFAR-10 result is the DDPM/DDIM cosine v-prediction U-Net:

```text
checkpoint: checkpoints/cifar10_unet_ddpm_cosine_vpred_500ep.pt
sample: figures/cifar_flow/final_candidate_cosine_vpred_raw_500ep_ddim250_cfg2.5_upscaled.png
metric json: results/cifar_flow/metrics_ddpm_cosine_vpred_raw_cfg25_5k.json
FID: 13.3722
Inception Score: 5.2858 +/- 0.1446
```

The strongest visual result so far is the 64x64 EDM Pets branch:

```text
binary cats-vs-dogs:
figures/visual64/pets_binary_edm_300ep_heun50_cfg2.png

37-class Pets:
figures/visual64/pets_imagefolder_edm_300ep_heun50_cfg2.png
```

The 64x64 branch is substantially clearer than the CIFAR-10 32x32 display images and should be used for README visual presentation.

## Main Technical Branches

### CIFAR-10 DDPM/DDIM

Purpose:

```text
Stable quantitative baseline with FID and Inception Score.
```

Main files:

```text
labs/lab_cifar_flow/train_cifar10_ddpm_unet.py
labs/lab_cifar_flow/sample_cifar10_ddpm.py
labs/lab_cifar_flow/eval_cifar10_ddpm_metrics.py
configs/cifar10_unet_ddpm_cosine_vpred_500ep.yaml
configs/cifar10_final_showcase_cosine_vpred.yaml
```

### CIFAR-10 EDM

Purpose:

```text
Quality upgrade for sharper CIFAR-10 generation using EDM preconditioning and Karras Heun sampling.
```

Current state:

```text
Initial EDM run reached an epoch-20 checkpoint.
Next run resumes from that checkpoint using:
configs/cifar10_unet_edm_resume_300ep.yaml
```

### Visual64 EDM

Purpose:

```text
README-friendly visual generation at 64x64.
```

Main files:

```text
labs/lab_visual64/train_visual64_edm_unet.py
labs/lab_visual64/sample_visual64_edm.py
configs/visual64_pets_imagefolder_edm_300ep.yaml
configs/visual64_pets_binary_edm_300ep.yaml
```

## Reproduction Commands

Regenerate CIFAR-10 final DDPM/DDIM showcase:

```bash
conda run -n fm_diffusion python labs/lab_cifar_flow/sample_cifar10_ddpm.py \
  --config configs/cifar10_final_showcase_cosine_vpred_upscaled.yaml
```

Regenerate final DDPM/DDIM metrics:

```bash
conda run -n fm_diffusion python labs/lab_cifar_flow/eval_cifar10_ddpm_metrics.py \
  --config configs/cifar10_metrics_final_cosine_vpred_5k.yaml
```

Regenerate binary Pets64 EDM showcase:

```bash
conda run -n fm_diffusion python labs/lab_visual64/sample_visual64_edm.py \
  --config configs/visual64_pets_binary_edm_sampling_300ep.yaml
```

Resume CIFAR-10 EDM quality training:

```bash
conda run -n fm_diffusion python labs/lab_cifar_flow/train_cifar10_edm_unet.py \
  --config configs/cifar10_unet_edm_resume_300ep.yaml
```

## Remaining Quality Work

The project is presentable now, but the active quality-upgrade goal is not fully closed until the resumed CIFAR-10 EDM run is sampled and compared against the DDPM/DDIM baseline.

Required next evidence:

```text
figures/cifar_flow/edm_cifar10_resume_300ep_heun40_cfg2_upscaled.png
results/cifar_flow/sample_config_edm_cifar10_resume_300ep_heun40_cfg2.json
optional: FID/IS for the resumed EDM checkpoint
```
