# Lab 5: Fast Sampling and Discrete Diffusion

## 1. Goal

This lab studies fast sampling, NFE comparison, and a toy discrete diffusion process.

## 2. Part A: NFE Comparison

Run:

```bash
bash labs/lab5/scripts/run_nfe_compare.sh
```

## 3. Part B: Mask Corruption

Run:

```bash
bash labs/lab5/scripts/run_mask_corruption.sh
```

## 4. Part C: Train Discrete Denoiser

Run:

```bash
bash labs/lab5/scripts/train_discrete_denoiser.sh
```

## 5. Part D: Sample Discrete Denoiser

Run:

```bash
bash labs/lab5/scripts/sample_discrete_denoiser.sh
```

## 6. Part E: Theory Figures

Run:

```bash
python labs/lab5/05_make_stage5_figures.py
```

## 7. Expected Outputs

- `figures/day5/nfe_compare.png`
- `figures/day5/mask_corruption_process.png`
- `figures/day5/discrete_reverse_process.png`
- `figures/day5/final_unified_framework.png`
- `figures/day5/continuous_vs_discrete_diffusion.png`
