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
