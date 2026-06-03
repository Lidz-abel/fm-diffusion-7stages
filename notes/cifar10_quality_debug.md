# CIFAR-10 Rectified Flow Quality Debug

## Phase 3 Goal

Phase 3 focuses on improving the CIFAR-10 U-Net Rectified Flow baseline before adding Mini-DiT, larger image resolution, reflow, or FlowDCN.

The target is not photorealism. The target is to move from blurry color fields to samples with clear CIFAR-10 color distribution, class-dependent structure, and rough object layout.

## 1. Debug Before Training More

Run:

```bash
python labs/lab_cifar_flow/debug_flow_tuple.py --device cuda --save_dir figures/cifar_flow/debug
```

Expected outputs:

- `figures/cifar_flow/debug/debug_real_images.png`
- `figures/cifar_flow/debug/debug_noise_images.png`
- `figures/cifar_flow/debug/debug_xt_interpolation.png`
- `figures/cifar_flow/debug/debug_velocity_stats.txt`

This checks whether real images, Gaussian noise, straight interpolation, target velocity, and image saving are all sane.

## 2. EMA

EMA is added through `src/ema.py` and `--use_ema`.

Training checkpoints now save:

- `model`: raw model weights
- `ema`: EMA model weights
- `step`
- `update_step`
- `epoch`
- `loss`
- `args`

Sampling supports:

```bash
python labs/lab_cifar_flow/sample_cifar10_fm.py --ckpt checkpoints/cifar10_unet_fm_v2.pt --use_ema --solver heun --nfe 100 --all_classes --num_per_class 8 --device cuda
```

Compare raw vs EMA:

```bash
python labs/lab_cifar_flow/compare_ema_raw.py --ckpt checkpoints/cifar10_unet_fm_v2.pt --device cuda
```

## 3. Attention U-Net

The current CIFAR U-Net already supports attention through:

```bash
--use_attention
```

In `configs/cifar10_unet_fm_v2.yaml`, this is enabled through:

```yaml
model:
  use_attention: true
  attention_resolutions: [16, 8]
```

## 4. Training Stability

The training script now supports:

- AdamW
- AMP
- EMA
- gradient accumulation
- warmup
- cosine learning-rate schedule
- config files

Short test:

```bash
python labs/lab_cifar_flow/train_cifar10_unet_fm.py --config configs/cifar10_unet_fm_v2.yaml --epochs 1 --max_steps 5 --batch_size 8 --base_channels 32 --time_dim 128 --device cuda
```

Official quality run:

```bash
python labs/lab_cifar_flow/train_cifar10_unet_fm.py --config configs/cifar10_unet_fm_v2.yaml --epochs 100 --device cuda
```

Longer baseline:

```bash
python labs/lab_cifar_flow/train_cifar10_unet_fm.py --config configs/cifar10_unet_fm_v2.yaml --epochs 300 --device cuda
```

## 5. Quality Sampling Setting

Use high-quality sampling for judging model quality:

```text
solver = heun
NFE = 100
cfg_scale = 1.0 or 2.0
use_ema = true
```

Low-NFE runs should be used for speed ablations, not for deciding whether the model is good.

## 6. Next Decision

If U-Net v2 with attention, EMA, 300 epochs, Heun NFE 100, and CFG scale 1 or 2 is still blurry, stop adding epochs and inspect the training objective:

- compare with DDPM or EDM-style noise prediction;
- test non-uniform time sampling or loss weighting;
- consider reflow only after the base model produces usable structures.

## 7. Time Sampling and Loss Weighting

The training code supports:

```bash
--time_sampling uniform
--time_sampling beta --time_beta_alpha 3.0 --time_beta_beta 1.0
--loss_weighting none
--loss_weighting data_end --loss_weight_lambda 1.0
--loss_weighting middle --loss_weight_lambda 1.0
```

The first direct 300-epoch quality experiment uses:

```text
time_sampling = beta(3, 1)
loss_weighting = data_end
loss_weight_lambda = 1.0
```

This biases training toward later times near the data endpoint, where image structure and detail matter most for visual quality.
