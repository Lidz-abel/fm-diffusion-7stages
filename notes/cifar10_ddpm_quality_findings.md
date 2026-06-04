# CIFAR-10 DDPM Quality Findings

## Current Best Visual Candidate

Current strongest early-stage preview:

```text
figures/cifar_flow/showcase_ddpm_linear_ep20_raw_ddim250_cfg2.png
```

Configuration:

```text
checkpoint: checkpoints/cifar10_unet_ddpm_strong_500ep.pt
weights: raw model weights
sampler: DDIM
DDIM steps: 250
CFG scale: 2.0
clip_x0: true
checkpoint age: around epoch 20
```

Visual observation:

- Airplane, automobile, horse, ship, and truck rows already show recognizable CIFAR-10 structure.
- Animal classes are still blurry but have stronger object-like silhouettes than the Rectified Flow baselines.
- This branch is currently the best candidate for README/showcase samples while training continues.

## EMA vs Raw Observation

At early checkpoints, raw DDPM weights are much better than EMA weights.

Observed files:

```text
figures/cifar_flow/preview_ddpm_linear_ep20_ddim100_cfg2.png
figures/cifar_flow/preview_ddpm_linear_ep20_raw_ddim100_cfg2.png
```

Interpretation:

- `ema_decay=0.9999` is too slow to track early training.
- EMA may become useful later, but early visual debugging should inspect raw weights too.
- For future runs, consider saving both raw and EMA samples at each evaluation point.
- If EMA remains worse in later checkpoints, test lower decay values such as `0.999` or `0.9995`.

## Cosine V-Prediction Branch

Current early preview:

```text
figures/cifar_flow/showcase_ddpm_cosine_vpred_ep10_raw_ddim250_cfg2.png
```

The cosine v-prediction branch is younger, around epoch 10 in this preview. It already shows object-like structure, but it is not yet a fair comparison with the linear epsilon branch at epoch 20.

## Next Actions

1. Continue both DDPM branches until at least epoch 100.
2. Generate raw and EMA previews at the same checkpoint age.
3. Run FID and Inception Score with `configs/cifar10_metrics_ddpm_*.yaml`.
4. Use the best visual branch for README samples and the best metric branch for the results table.

## Epoch 40 Linear DDPM Update

Updated preview files:

```text
figures/cifar_flow/showcase_ddpm_linear_ep40_raw_ddim250_cfg2.png
figures/cifar_flow/showcase_ddpm_linear_ep40_ema_ddim250_cfg2.png
figures/cifar_flow/cfgscan_ddpm_linear_ep40_raw_ddim250_cfg1.0.png
figures/cifar_flow/cfgscan_ddpm_linear_ep40_raw_ddim250_cfg1.5.png
figures/cifar_flow/cfgscan_ddpm_linear_ep40_raw_ddim250_cfg2.5.png
figures/cifar_flow/cfgscan_ddpm_linear_ep40_raw_ddim250_cfg3.0.png
```

Findings:

- Raw weights remain much better than EMA at this point.
- EMA still produces color-texture patterns instead of coherent CIFAR-10 objects.
- `cfg_scale=2.5` is the best balanced visual setting so far.
- `cfg_scale=3.0` strengthens class identity for airplane/automobile/ship/truck, but introduces more guidance artifacts.
- Current recommended showcase candidate:

```text
figures/cifar_flow/cfgscan_ddpm_linear_ep40_raw_ddim250_cfg2.5.png
```

This is now the first sample grid that is clearly suitable for an intermediate README progress section.

## 500 Epoch Final Candidate Update

Final candidate files:

```text
figures/cifar_flow/final_candidate_linear_raw_500ep_ddim250_cfg2.5.png
figures/cifar_flow/final_candidate_cosine_vpred_raw_500ep_ddim250_cfg2.5.png
```

Findings:

- Both branches are clearly better than the earlier Rectified Flow results.
- The linear epsilon model gives strong class structure, especially for airplane, automobile, ship, truck, horse, and frog.
- The cosine v-prediction model is the current best visual candidate. It has more natural-looking animals and stronger object separation while preserving clear vehicles and ships.
- Current main showcase branch:

```text
checkpoint: checkpoints/cifar10_unet_ddpm_cosine_vpred_500ep.pt
weights: raw
sampler: DDIM
DDIM steps: 250
CFG scale: 2.5
sample grid: figures/cifar_flow/final_candidate_cosine_vpred_raw_500ep_ddim250_cfg2.5.png
sample config: configs/cifar10_final_showcase_cosine_vpred.yaml
```

## Final 5k Metric Result

Final metric files:

```text
results/cifar_flow/metrics_ddpm_cosine_vpred_raw_cfg25_5k.json
results/cifar_flow/metrics_ddpm_final.csv
figures/cifar_flow/metrics_ddpm_cosine_vpred_raw_cfg25_5k_preview.png
```

Measured configuration:

```text
checkpoint: checkpoints/cifar10_unet_ddpm_cosine_vpred_500ep.pt
weights: raw
sampler: DDIM
DDIM steps: 250
CFG scale: 2.5
schedule: cosine
prediction type: v_prediction
num samples: 5000
real split: CIFAR-10 train
pretrained Inception: true
metric config: configs/cifar10_metrics_final_cosine_vpred_5k.yaml
```

Final metrics:

```text
FID: 13.3722
Inception Score: 5.2858 +/- 0.1446
seconds total: 1520.69
seconds per sample: 0.3041
```

Conclusion:

- The cosine v-prediction DDPM branch is the current best project-quality model.
- It is suitable as the main README/showcase result for the CIFAR-10 image generation upgrade.
- Rectified Flow remains useful for the learning-project narrative and sampler analysis, but it is not the strongest visual branch at the current training budget.
- Further quality work should only target high-leverage changes such as a stronger U-Net, EDM-style preconditioning, or better augmentation/training schedules. Broad additional ablations are not needed until there is a new candidate likely to beat this result.
```
