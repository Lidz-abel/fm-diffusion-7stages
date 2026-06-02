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

Compares NFE 5, 10, 20, 50, and 100 with Euler sampling. Outputs:

- `figures/cifar_flow/nfe_ablation_unet.png`
- `results/cifar_flow/nfe_ablation.csv`

### Solver Ablation

Compares Euler and Heun ODE sampling at NFE 20. Outputs:

- `figures/cifar_flow/solver_ablation_unet.png`
- `results/cifar_flow/solver_ablation.csv`

### Guidance Ablation

Compares CFG-like velocity guidance scales 0, 1, 2, and 4 with Heun at NFE 50. Outputs:

- `figures/cifar_flow/guidance_ablation_unet.png`
- `results/cifar_flow/guidance_ablation.csv`

### Runtime / Memory Benchmark

Records sampling time and memory data in CSV files under `results/cifar_flow/`.

- `figures/cifar_flow/runtime_vs_nfe.png`
- `results/cifar_flow/runtime_summary.csv`
- `results/cifar_flow/memory_summary.csv`

### Failure Cases

Low NFE, under-trained checkpoints, and large guidance scales are the main failure modes to document when comparing generated images.
