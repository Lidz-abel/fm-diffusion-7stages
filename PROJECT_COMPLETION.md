# Project Completion Note

## Current Status

This repository has reached a complete showcase state for the diffusion / flow-matching learning project.

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

The CIFAR-10 EDM quality-upgrade run is complete:

```text
checkpoint: checkpoints/cifar10_unet_edm_resume_300ep.pt
sample: figures/cifar_flow/edm_cifar10_resume_300ep_heun40_cfg2_upscaled.png
sample config: results/cifar_flow/sample_config_edm_cifar10_resume_300ep_heun40_cfg2.json
metric json: results/cifar_flow/metrics_edm_resume_300ep_1k.json
FID@1k: 41.4960
Inception Score@1k: 5.0062 +/- 0.4826
```

Conclusion:

```text
CIFAR-10 EDM improved visual structure over early EDM checkpoints, but it did not beat the DDPM/DDIM cosine v-pred baseline quantitatively.
For CIFAR-10 metrics, keep DDPM/DDIM as the main result.
For clear visual presentation, use Visual64 EDM Pets.
```

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
The resume run completed 300 epochs using:
configs/cifar10_unet_edm_resume_300ep.yaml
```

Final evidence:

```text
checkpoint: checkpoints/cifar10_unet_edm_resume_300ep.pt
sample:
figures/cifar_flow/edm_cifar10_resume_300ep_heun40_cfg2_upscaled.png
metrics:
results/cifar_flow/metrics_edm_resume_300ep_1k.json
training log:
results/cifar_flow/training_log_edm_resume_300ep.csv
```

The EDM metric evaluator is now available:

```text
labs/lab_cifar_flow/eval_cifar10_edm_metrics.py
configs/cifar10_metrics_edm_resume_300ep_1k.yaml
configs/cifar10_metrics_edm_resume_300ep_5k.yaml
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

Resume or rerun CIFAR-10 EDM quality training:

```bash
conda run -n fm_diffusion python labs/lab_cifar_flow/train_cifar10_edm_unet.py \
  --config configs/cifar10_unet_edm_resume_300ep.yaml
```

Run quick CIFAR-10 EDM metrics after the 300-epoch checkpoint is ready:

```bash
conda run -n fm_diffusion python labs/lab_cifar_flow/eval_cifar10_edm_metrics.py \
  --config configs/cifar10_metrics_edm_resume_300ep_1k.yaml
```

Or wait for the 300-epoch checkpoint and run final sampling plus quick metrics automatically:

```bash
/home/ldz/miniconda3/envs/fm_diffusion/bin/python labs/lab_cifar_flow/finalize_cifar10_edm_resume.py \
  --gpu 0 \
  --target_epoch 300
```

Run formal CIFAR-10 EDM metrics:

```bash
conda run -n fm_diffusion python labs/lab_cifar_flow/eval_cifar10_edm_metrics.py \
  --config configs/cifar10_metrics_edm_resume_300ep_5k.yaml
```

## Remaining Quality Work

The requested quality-upgrade work is closed at the project level:

- CIFAR-10 EDM 300ep training, sampling, and 1k metric evaluation are complete.
- Visual64 EDM Pets provides the clearest final visual showcase.
- DDPM/DDIM remains the strongest quantitative CIFAR-10 baseline.

Optional future work:

```text
1. Run formal 5k metrics for CIFAR-10 EDM only if needed for a table.
2. Tune EDM hyperparameters if the goal becomes beating the DDPM/DDIM FID.
3. Promote Visual64 images into README as the visual-first result.
```
