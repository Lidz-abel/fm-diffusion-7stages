# CIFAR-10 Rectified Flow and DiT Report

## Phase 1: CIFAR-10 U-Net Rectified Flow Baseline

This phase implements a class-conditional U-Net velocity network trained with the straight Rectified Flow path.

Deliverables:

- `src/cifar10_dataset.py`
- `src/image_flow_matching.py`
- `src/image_unet.py`
- `src/image_flow_samplers.py`
- `labs/lab_cifar_flow/train_cifar10_unet_fm.py`
- `labs/lab_cifar_flow/sample_cifar10_fm.py`

## Phase 2: Sampling Analysis

### NFE Ablation

Compares NFE 5, 10, 20, 50, and 100.

### Solver Ablation

Compares Euler and Heun ODE sampling.

### Guidance Ablation

Compares CFG-like velocity guidance scales 0, 1, 2, and 4.

### Runtime / Memory Benchmark

Records sampling time and memory data in CSV files under `results/cifar_flow/`.

### Failure Cases

Low NFE, under-trained checkpoints, and large guidance scales should be documented with example figures after full training.
