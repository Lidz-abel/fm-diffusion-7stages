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
