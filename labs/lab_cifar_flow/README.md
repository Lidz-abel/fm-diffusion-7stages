# CIFAR-10 Rectified Flow Lab

This lab trains and evaluates a class-conditional U-Net Rectified Flow baseline on CIFAR-10.

## Phase 1: Baseline

Train a small smoke run:

```bash
python labs/lab_cifar_flow/train_cifar10_unet_fm.py --epochs 1 --max_steps 5 --batch_size 16 --base_channels 32 --time_dim 128 --device cpu
```

Train the intended baseline:

```bash
python labs/lab_cifar_flow/train_cifar10_unet_fm.py --epochs 100 --batch_size 256 --base_channels 128 --time_dim 512 --device cuda
```

Sample all classes:

```bash
python labs/lab_cifar_flow/sample_cifar10_fm.py --all_classes --solver euler --nfe 50 --num_per_class 8 --device cuda
```

## Phase 2: Sampling Analysis

```bash
python labs/lab_cifar_flow/eval_nfe_ablation.py --ckpt checkpoints/cifar10_unet_fm.pt --solver euler --nfe_list 5 10 20 50 100 --num_per_class 8 --device cuda
python labs/lab_cifar_flow/eval_solver_ablation.py --ckpt checkpoints/cifar10_unet_fm.pt --solvers euler heun --nfe 20 --num_per_class 8 --device cuda
python labs/lab_cifar_flow/eval_guidance_ablation.py --ckpt checkpoints/cifar10_unet_fm.pt --solver heun --nfe 50 --cfg_scales 0 1 2 4 --num_per_class 8 --device cuda
python labs/lab_cifar_flow/benchmark_sampling.py --ckpt checkpoints/cifar10_unet_fm.pt --solvers euler heun --nfe_list 5 10 20 50 100 --device cuda
```

## Phase 3: Quality Upgrade

Debug the data path:

```bash
python labs/lab_cifar_flow/debug_flow_tuple.py --device cuda --save_dir figures/cifar_flow/debug
```

Run an Attention U-Net + EMA smoke test:

```bash
python labs/lab_cifar_flow/train_cifar10_unet_fm.py --config configs/cifar10_unet_fm_v2.yaml --epochs 1 --max_steps 5 --batch_size 8 --base_channels 32 --time_dim 128 --device cuda
```

Sample the EMA model with a high-quality setting:

```bash
python labs/lab_cifar_flow/sample_cifar10_fm.py --ckpt checkpoints/cifar10_unet_fm_v2.pt --use_ema --solver heun --nfe 100 --all_classes --num_per_class 8 --device cuda
```

Compare raw and EMA weights:

```bash
python labs/lab_cifar_flow/compare_ema_raw.py --ckpt checkpoints/cifar10_unet_fm_v2.pt --device cuda
```

## Phase 3.5: DDPM Strong Baselines

The Rectified Flow branch remains the main project line, but the DDPM branch is used as a strong image-quality reference.

Train the linear beta, epsilon-prediction DDPM baseline:

```bash
python labs/lab_cifar_flow/train_cifar10_ddpm_unet.py \
  --config configs/cifar10_unet_ddpm_strong_500ep.yaml \
  --device cuda
```

Train the stronger cosine-schedule, v-prediction DDPM baseline:

```bash
python labs/lab_cifar_flow/train_cifar10_ddpm_unet.py \
  --config configs/cifar10_unet_ddpm_cosine_vpred_500ep.yaml \
  --device cuda
```

Sample the linear DDPM baseline with DDIM:

```bash
python labs/lab_cifar_flow/sample_cifar10_ddpm.py \
  --config configs/cifar10_ddim_sampling_strong.yaml
```

Sample the cosine v-prediction baseline with DDIM:

```bash
python labs/lab_cifar_flow/sample_cifar10_ddpm.py \
  --config configs/cifar10_ddim_sampling_cosine_vpred.yaml
```

Evaluate FID and Inception Score for the linear DDPM baseline:

```bash
python labs/lab_cifar_flow/eval_cifar10_ddpm_metrics.py \
  --config configs/cifar10_metrics_ddpm_strong.yaml
```

Evaluate FID and Inception Score for the cosine v-prediction baseline:

```bash
python labs/lab_cifar_flow/eval_cifar10_ddpm_metrics.py \
  --config configs/cifar10_metrics_ddpm_cosine_vpred.yaml
```

## Current Final Showcase Candidate

The current best visual and metric branch is:

```text
cosine schedule + v-prediction + Attention U-Net + raw weights + DDIM 250 + CFG 2.5
```

Final 5k-sample metrics:

```text
FID: 13.3722
Inception Score: 5.2858 +/- 0.1446
```

Regenerate the final showcase grid:

```bash
python labs/lab_cifar_flow/sample_cifar10_ddpm.py \
  --config configs/cifar10_final_showcase_cosine_vpred.yaml
```

Regenerate the README-friendly nearest-neighbor upscaled grid:

```bash
python labs/lab_cifar_flow/sample_cifar10_ddpm.py \
  --config configs/cifar10_final_showcase_cosine_vpred_upscaled.yaml
```

Run the 5k-sample final metric evaluation:

```bash
python labs/lab_cifar_flow/eval_cifar10_ddpm_metrics.py \
  --config configs/cifar10_metrics_final_cosine_vpred_5k.yaml
```

Current output targets:

```text
figures/cifar_flow/final_candidate_cosine_vpred_raw_500ep_ddim250_cfg2.5.png
figures/cifar_flow/final_candidate_cosine_vpred_raw_500ep_ddim250_cfg2.5_upscaled.png
figures/cifar_flow/metrics_ddpm_cosine_vpred_raw_cfg25_5k_preview.png
results/cifar_flow/metrics_ddpm_cosine_vpred_raw_cfg25_5k.json
results/cifar_flow/metrics_ddpm_final.csv
```

Each run writes:

- checkpoint under `checkpoints/`
- launch log under `logs/`
- resolved runtime config under `results/cifar_flow/config_used_*.json`
- training loss CSV under `results/cifar_flow/training_log_*.csv`
- sample or metric outputs under `figures/cifar_flow/` and `results/cifar_flow/`

## Phase 4: EDM Quality Upgrade

The next quality upgrade is EDM-style preconditioned denoising with Karras sigma sampling. This branch is designed to improve sharpness beyond the current DDPM/DDIM result.

Run a smoke test:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n fm_diffusion python labs/lab_cifar_flow/train_cifar10_edm_unet.py \
  --config configs/cifar10_unet_edm_smoke.yaml \
  --max_steps 5
```

Train the main CIFAR-10 EDM model:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n fm_diffusion python labs/lab_cifar_flow/train_cifar10_edm_unet.py \
  --config configs/cifar10_unet_edm_500ep.yaml
```

Sample the EDM model:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n fm_diffusion python labs/lab_cifar_flow/sample_cifar10_edm.py \
  --config configs/cifar10_edm_sampling_500ep.yaml
```

Expected outputs:

```text
checkpoints/cifar10_unet_edm_500ep.pt
figures/cifar_flow/edm_cifar10_500ep_heun40_cfg2_upscaled.png
results/cifar_flow/sample_config_edm_cifar10_500ep_heun40_cfg2.json
```
