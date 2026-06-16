# Stage 8B: Generic Class-Conditional Image CFG

## Goal

Stage 8B prepares the project for ImageNet-128 class-based classifier-free guidance by removing the most important CIFAR-specific assumptions from the sampling and smoke-training path.

The immediate target is not ImageNet training quality. The target is to verify:

```text
image_size: 32 / 64 / 128
num_classes: 10 / 37 / 1000
null_label = num_classes
DDPM/DDIM + CFG sampling without hard-coded 32x32 shape
one-step training-loop smoke test
```

## New Files

```text
src/class_cond_unet.py
src/image_dataset.py
src/imagenet_dataset.py

labs/lab_image_cfg/README.md
labs/lab_image_cfg/debug_shape_test.py
labs/lab_image_cfg/sample_class_cond_ddpm_cfg.py
labs/lab_image_cfg/train_class_cond_ddpm_unet.py

configs/class_cond_ddpm_unet_template.yaml
configs/cifar10_class_cond_ddpm_sampling.yaml
configs/imagenet128_subset10_unet_ddpm_cosine_vpred.yaml
configs/imagenet128_subset10_sampling.yaml
```

## Design

`ClassConditionalUNet` inherits the existing `CIFAR10FlowUNet` implementation directly. This is intentional: it keeps the old checkpoint `state_dict` keys compatible while giving the project a generic model name for 32x32, 64x64, and 128x128 class-conditional image generation.

The generic dataset entry supports:

```text
cifar10
imagefolder
synthetic
```

The `synthetic` dataset exists only for smoke tests and avoids requiring ImageNet data before verifying model shape and training-loop correctness.

## GPU Usage

The machine had several mostly idle RTX 3090 cards. This stage used at most two GPUs:

```text
GPU 1: shape tests and CIFAR generic sampling
GPU 3: 128x128 / 1000-class shape test and training smoke
```

## Validation Commands

Environment:

```bash
/home/ldz/miniconda3/envs/fm_diffusion/bin/python
```

### 32x32 / 10-class Shape Test

```bash
CUDA_VISIBLE_DEVICES=1 /home/ldz/miniconda3/envs/fm_diffusion/bin/python \
  labs/lab_image_cfg/debug_shape_test.py \
  --image_size 32 \
  --num_classes 10 \
  --null_label 10 \
  --batch_size 2 \
  --base_channels 16 \
  --time_dim 64 \
  --attention_resolutions 16 8 \
  --device cuda
```

Result:

```text
input: (2, 3, 32, 32)
conditional output: (2, 3, 32, 32)
unconditional output: (2, 3, 32, 32)
shape test passed
```

### 64x64 / 37-class Shape Test

```bash
CUDA_VISIBLE_DEVICES=1 /home/ldz/miniconda3/envs/fm_diffusion/bin/python \
  labs/lab_image_cfg/debug_shape_test.py \
  --image_size 64 \
  --num_classes 37 \
  --null_label 37 \
  --batch_size 2 \
  --base_channels 16 \
  --time_dim 64 \
  --attention_resolutions 32 16 \
  --device cuda
```

Result:

```text
input: (2, 3, 64, 64)
conditional output: (2, 3, 64, 64)
unconditional output: (2, 3, 64, 64)
shape test passed
```

### 128x128 / 1000-class Shape Test

```bash
CUDA_VISIBLE_DEVICES=3 /home/ldz/miniconda3/envs/fm_diffusion/bin/python \
  labs/lab_image_cfg/debug_shape_test.py \
  --image_size 128 \
  --num_classes 1000 \
  --null_label 1000 \
  --batch_size 2 \
  --base_channels 16 \
  --time_dim 64 \
  --attention_resolutions 32 16 \
  --device cuda
```

Result:

```text
input: (2, 3, 128, 128)
conditional output: (2, 3, 128, 128)
unconditional output: (2, 3, 128, 128)
shape test passed
```

### Generic Training Smoke

```bash
CUDA_VISIBLE_DEVICES=3 /home/ldz/miniconda3/envs/fm_diffusion/bin/python \
  labs/lab_image_cfg/train_class_cond_ddpm_unet.py \
  --config configs/class_cond_ddpm_unet_template.yaml \
  --max_steps 1
```

Outputs:

```text
checkpoints/class_cond_ddpm_smoke.pt
results/image_cfg/training_log_smoke.csv
results/image_cfg/config_used_smoke.json
```

### Generic CIFAR-10 DDIM Sampling

```bash
CUDA_VISIBLE_DEVICES=1 /home/ldz/miniconda3/envs/fm_diffusion/bin/python \
  labs/lab_image_cfg/sample_class_cond_ddpm_cfg.py \
  --config configs/cifar10_class_cond_ddpm_sampling.yaml
```

Outputs:

```text
figures/image_cfg/cifar10_generic_ddim20_cfg2.5.png
results/image_cfg/sample_config_cifar10_generic_ddim20_cfg2.5.json
```

The generated sample grid is a short DDIM-20 smoke result from the existing CIFAR-10 cosine v-prediction checkpoint. It proves that the new sampler can load the old checkpoint and infer the correct sample shape without hard-coding `32x32`.

## Completion Criteria

Stage 8B is complete at the code-smoke level:

```text
1. New model entry supports image_size = 32 / 64 / 128.
2. New model entry supports num_classes = 10 / 37 / 1000.
3. null_label = num_classes works for conditional and unconditional forward passes.
4. Generic sampler no longer hard-codes shape = (B, 3, 32, 32).
5. Generic training script runs one DDPM loss/backward/update step.
6. Existing CIFAR-10 checkpoint can be sampled through the new generic script.
7. ImageNet-128 subset10 configs exist for the next stage.
```

## Next Step

Proceed to Stage 8C:

```text
ImageNet-128 subset10 data preparation and sanity training.
```

The next stage should not start with full ImageNet-1k. It should first create an ImageFolder-compatible subset containing the target Figure-11 class, such as `corgi dog`, then run a small 10-class training job and compare `w=0` vs `w=4`.
