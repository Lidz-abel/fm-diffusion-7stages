# Stage 8B: Generic Class-Conditional Image CFG

This lab generalizes the current CIFAR / Visual64 class-conditional diffusion code toward 128x128 ImageNet-style CFG experiments.

## Smoke Tests

Shape test for ImageNet-128 scale:

```bash
CUDA_VISIBLE_DEVICES=1 python labs/lab_image_cfg/debug_shape_test.py \
  --image_size 128 \
  --num_classes 1000 \
  --null_label 1000 \
  --batch_size 2 \
  --base_channels 32 \
  --time_dim 128 \
  --attention_resolutions 32 16 \
  --device cuda
```

Generic DDPM/DDIM sampling with the existing CIFAR-10 checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 python labs/lab_image_cfg/sample_class_cond_ddpm_cfg.py \
  --config configs/cifar10_class_cond_ddpm_sampling.yaml
```

One-step generic training smoke test:

```bash
CUDA_VISIBLE_DEVICES=3 python labs/lab_image_cfg/train_class_cond_ddpm_unet.py \
  --config configs/class_cond_ddpm_unet_template.yaml \
  --max_steps 1
```
