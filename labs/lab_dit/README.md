# Stage 4.5: Mini-DiT Lab

## Goal

Implement a minimal DiT denoiser for MNIST diffusion and compare it with the Stage 4 U-Net diffusion model.

## Train

```bash
python labs/lab_dit/train_mnist_dit.py --device cuda --epochs 20
```

## Sample

```bash
python labs/lab_dit/sample_mnist_dit.py --class_id 3 --cfg_scale 4 --device cuda
```

## Compare U-Net and DiT

```bash
python labs/lab_dit/compare_unet_dit.py --device cuda
```
