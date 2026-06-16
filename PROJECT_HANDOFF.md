# Project Handoff Note

## 1. Current Project State

This repository is now a complete diffusion / flow-matching learning and showcase project.

The completed parts are:

- Course labs: ODE/SDE, Flow Matching, DDPM, CFG, Mini-DiT, discrete diffusion.
- CIFAR-10 experiments: Rectified Flow, NFE / solver / guidance ablation, DDPM/DDIM, EDM.
- Visual64 experiments: Oxford-IIIT Pets EDM branch with the strongest visual results.
- Documentation:
  - `README.md`
  - `PROJECT_COMPLETION.md`
  - `PROJECT_STAGE_SUMMARY.tex`
  - `PROJECT_STAGE_SUMMARY.pdf`

Best current results:

```text
CIFAR-10 quantitative best:
checkpoint: checkpoints/cifar10_unet_ddpm_cosine_vpred_500ep.pt
figure: figures/cifar_flow/final_candidate_cosine_vpred_raw_500ep_ddim250_cfg2.5_upscaled.png
metric: results/cifar_flow/metrics_ddpm_cosine_vpred_raw_cfg25_5k.json
FID: 13.3722
IS: 5.2858 +/- 0.1446

Visual best:
figures/visual64/final_visual64_showcase_panel.png
figures/visual64/visual64_best_binary_pets_heun80_cfg2.png
figures/visual64/visual64_best_37class_pets_heun80_cfg2.png
```

## 2. Active Goal

The recently completed next-stage goal is:

```text
Complete Stage 8A: Visual64 / CIFAR Figure-11-style CFG reproduction.
```

Stage 8A is the small-scale bridge toward ImageNet-128 class-based classifier-free guidance.

Target idea:

```text
same checkpoint
same class ids
same seed
same sampler
same number of steps

left:  cfg_scale = 0
right: cfg_scale = 4
```

Expected final artifact:

```text
figures/visual64/cfg_w0_vs_w4_panel.png
```

The figure should imitate the Figure 11 logic:

```text
Left: weak / no classifier-free guidance.
Right: classifier-free guidance with w = 4.
Conclusion: stronger CFG should improve class adherence, but may reduce diversity.
```

## 3. Current Stage 8A Status

Completed:

- Added the panel-making script:

```text
labs/lab_visual64/make_cfg_comparison_panel.py
```

- Verified syntax:

```bash
python -m py_compile labs/lab_visual64/make_cfg_comparison_panel.py
```

- Verified CLI:

```bash
python labs/lab_visual64/make_cfg_comparison_panel.py --help
```

- Confirmed the current sandbox had no usable CUDA:

```text
torch.cuda.is_available() = False
nvidia-smi cannot communicate with the NVIDIA driver
```

- Ran a small CPU preview for `cfg_scale=0.0`:

```text
figures/visual64/cfg_w0_preview_cpu.png
results/visual64/sample_config_cfg_w0_preview_cpu.json
```

- Generated the full `w=0` Visual64 sample grid:

```text
figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg0.png
```

- Generated the full `w=4` Visual64 sample grid:

```text
figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg4.png
```

- Built the final side-by-side panel:

```text
figures/visual64/cfg_w0_vs_w4_panel.png
```

- Added Stage 8A report:

```text
reports/visual64_cfg_comparison.md
```

Stage 8A is now complete. The commands below remain as the recommended GPU reproduction path.

## 4. GPU Reproduction Commands for Stage 8A

Use the existing Visual64 Pets ImageFolder EDM checkpoint:

```text
checkpoints/visual64_pets_imagefolder_unet_edm_300ep.pt
```

Use the existing sampling config:

```text
configs/visual64_pets_imagefolder_edm_sampling_300ep.yaml
```

Recommended GPU command:

```bash
CUDA_VISIBLE_DEVICES=<free_gpu_id> python labs/lab_visual64/eval_visual64_sampling_sweep.py \
  --config configs/visual64_pets_imagefolder_edm_sampling_300ep.yaml \
  --tag pets_cfg_figure \
  --class_ids 0 1 2 3 4 5 6 7 \
  --num_per_class 4 \
  --cfg_scales 0.0 4.0 \
  --num_steps_list 80 \
  --seed 123 \
  --device cuda
```

Expected output:

```text
figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg0.png
figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg4.png
figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_sampling_sweep_panel.png
results/visual64/sampling_sweep/pets_cfg_figure_sampling_sweep.json
```

Then build the Figure-11-style panel:

```bash
python labs/lab_visual64/make_cfg_comparison_panel.py \
  --left figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg0.png \
  --right figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg4.png \
  --class_name "Oxford-IIIT Pets classes" \
  --left_title "without classifier-free guidance, w=0" \
  --right_title "classifier-free guidance, w=4" \
  --save_path figures/visual64/cfg_w0_vs_w4_panel.png \
  --metadata_out results/visual64/cfg_w0_vs_w4_panel.json
```

Expected output:

```text
figures/visual64/cfg_w0_vs_w4_panel.png
results/visual64/cfg_w0_vs_w4_panel.json
```

## 5. Stage 8A Report

The completed report is:

```text
reports/visual64_cfg_comparison.md
```

## 6. Stage 8A Completion Criteria

Stage 8A is complete because all of the following exist:

```text
labs/lab_visual64/make_cfg_comparison_panel.py
figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg0.png
figures/visual64/sampling_sweep/pets_cfg_figure/pets_cfg_figure_steps80_cfg4.png
figures/visual64/cfg_w0_vs_w4_panel.png
results/visual64/sampling_sweep/pets_cfg_figure_sampling_sweep.json
results/visual64/cfg_w0_vs_w4_panel.json
reports/visual64_cfg_comparison.md
```

And the final panel must visually show:

```text
left side: w=0
right side: w=4
same class ids
same seed
same checkpoint
same sampler
same steps
```

## 7. Stage 8B Status

Stage 8B has been completed at the code-smoke level.

Completed files:

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

reports/stage8b_generic_image_cfg.md
```

Verified behavior:

```text
image_size = 32 / 64 / 128
num_classes = 10 / 37 / 1000
null_label = num_classes
generic DDPM/DDIM CFG sampling without hard-coded 32x32 shape
one-step generic DDPM training smoke
old CIFAR-10 checkpoint sampled through the generic script
```

GPU usage:

```text
GPU 1: shape tests and CIFAR generic sampling
GPU 3: 128x128 / 1000-class shape test and training smoke
```

Representative outputs:

```text
figures/image_cfg/cifar10_generic_ddim20_cfg2.5.png
results/image_cfg/sample_config_cifar10_generic_ddim20_cfg2.5.json
checkpoints/class_cond_ddpm_smoke.pt
results/image_cfg/training_log_smoke.csv
results/image_cfg/config_used_smoke.json
```

Stage 8B report:

```text
reports/stage8b_generic_image_cfg.md
```

## 8. Next Stage

Next planned stage:

```text
Stage 8C: ImageNet-128 subset10 data preparation and sanity training.
```

The next step is to prepare an ImageFolder-compatible subset containing the target Figure-11 class, such as `corgi dog`, then train a 10-class 128x128 model and compare `w=0` vs `w=4`.

## 9. Notes for the Next Agent

- Do not delete existing checkpoints, logs, results, or figures.
- Preserve every config used for each experiment.
- If running a long GPU job, use `tmux` and log to `logs/`.
- Stage 8A has been completed; do not regress or overwrite its final panel unless intentionally rerunning the same comparison.
- Stage 8B has been completed with smoke tests; do not start ImageNet-1k before Stage 8C subset10 is working.
