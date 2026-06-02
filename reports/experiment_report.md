# Experiment Report: Day 1-5

## 1. Overview

This report summarizes the completed Day 1-5 experiments in the 7-stage Flow Matching, Diffusion Models, and FlowDCN learning project.

The completed scope covers:

- Day 1: toy data, Brownian motion, ODE solver, SDE solver, probability path visualization
- Day 2: 2D Flow Matching, ODE sampling, NFE comparison, vector field visualization
- Day 3: 2D DDPM / score matching, forward noising, reverse sampling, FM vs DDPM comparison
- Day 4: MNIST class-conditional diffusion with classifier-free guidance
- Day 5: fast sampling summary, toy discrete diffusion, unified theory figures

## 2. Day 1: ODE, SDE, and Probability Paths

### Code

Implemented:

- `src/toy_data.py`
- `src/ode.py`
- `src/sde.py`
- `src/visualization.py`
- `labs/lab1/run_probability_path.py`
- `labs/lab1/run_ode_sde.py`

### Experiments

Generated:

- `figures/stage1/probability_path_t_0.00.png`
- `figures/stage1/probability_path_t_0.25.png`
- `figures/stage1/probability_path_t_0.50.png`
- `figures/stage1/probability_path_t_0.75.png`
- `figures/stage1/probability_path_t_1.00.png`
- `figures/stage1/brownian_motion.png`
- `figures/stage1/ode_trajectories.png`
- `figures/stage1/sde_trajectories.png`

### Summary

Day 1 established the basic simulation tools. ODE dynamics are deterministic after the initial state is fixed, while SDE dynamics remain stochastic because of Brownian noise. Euler and Euler-Maruyama solvers provide the numerical foundation used by later flow and diffusion experiments.

## 3. Day 2: 2D Flow Matching

### Code

Implemented:

- `src/mlp.py`
- `src/flow_matching.py`
- `src/samplers.py`
- `labs/lab2/run_2d_flow_matching.py`

### Experiments

Generated:

- `figures/stage2/target_data.png`
- `figures/stage2/initial_noise.png`
- `figures/stage2/fm_samples_nfe_5.png`
- `figures/stage2/fm_samples_nfe_10.png`
- `figures/stage2/fm_samples_nfe_20.png`
- `figures/stage2/fm_samples_nfe_50.png`
- `figures/stage2/fm_samples_nfe_100.png`
- `figures/stage2/fm_samples_nfe_compare.png`
- `figures/stage2/fm_trajectories.png`
- `figures/stage2/fm_vector_field_t_0.00.png`
- `figures/stage2/fm_vector_field_t_0.25.png`
- `figures/stage2/fm_vector_field_t_0.50.png`
- `figures/stage2/fm_vector_field_t_0.75.png`

### Summary

Flow Matching uses the straight-line path

$$
x_t=(1-t)x_0+tz
$$

with target velocity

$$
u_t=z-x_0.
$$

Training is simulation-free because the model regresses directly on sampled path points and target velocities. Sampling requires solving an ODE from Gaussian noise to the learned data distribution. Larger NFE generally improves integration quality.

## 4. Day 3: 2D DDPM / Score Matching

### Code

Implemented:

- `src/diffusion.py`
- `labs/lab2/run_2d_ddpm.py`
- `notes/noise_score_velocity_table.md`

### Experiments

Generated:

- `figures/stage3/forward_noising.png`
- `figures/stage3/ddpm_samples.png`
- `figures/stage3/ddpm_sampling_trajectory.png`
- `figures/stage3/fm_vs_ddpm.png`

### Summary

DDPM trains a model to predict the injected Gaussian noise:

$$
\epsilon_\theta(x_t,t)\approx\epsilon.
$$

The forward marginal can be sampled in one step:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

Sampling starts from Gaussian noise and iteratively denoises. Unlike Flow Matching, DDPM sampling is a reverse Markov chain rather than direct ODE integration.

## 5. Day 4: MNIST Conditional Diffusion with CFG

### Code Acceptance

Status:

- `src/time_embedding.py` is independently testable.
- `src/unet.py` produces correct input/output shape: `[B, 1, 28, 28] -> [B, 1, 28, 28]`.
- `src/diffusion.py` supports MNIST conditional DDPM training through `ddpm_conditional_loss`.
- `src/cfg_sampler.py` implements classifier-free guidance.
- `labs/lab4/train_mnist_cfg.py` can train and save `checkpoints/mnist_cfg_unet.pt`.
- `labs/lab4/eval_cfg_scales.py` can generate CFG scale comparison figures.

### Experiment Acceptance

Generated:

- `figures/day4/samples_class3_scale4.png`
- `figures/day4/class_cond_samples.png`
- `figures/day4/cfg_scale_comparison.png`
- `figures/day4/latent_diffusion_pipeline.png`

The implemented CFG formula is:

$$
\epsilon_{\mathrm{cfg}}
=
\epsilon_{\mathrm{uncond}}
+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
$$

Interpretation:

- `scale = 0`: unconditional generation
- `scale = 1`: standard conditional generation
- `scale = 2/4`: stronger class conditioning
- `scale = 7`: strongest conditioning among tested settings, with possible diversity reduction or artifacts

### Report Acceptance

Available documentation:

- `notes/stage4_guidance_architecture.md`
- `notes/day4_guidance_architecture.md`
- `notebooks/day4_mnist_cfg.ipynb`
- `figures/day4/cfg_scale_comparison.png`

The notebook contains sections for MNIST data visualization, forward noising, U-Net explanation, CFG scale comparison, and U-Net / DiT / latent diffusion comparison.

## 6. Day 5: Fast Sampling and Discrete Diffusion

### Code

Implemented:

- `src/fast_sampling.py`
- `src/discrete_diffusion.py`
- `src/discrete_dataset.py`
- `src/discrete_denoiser.py`
- `src/theory_plotting.py`
- `labs/lab5/01_run_nfe_compare.py`
- `labs/lab5/02_run_mask_corruption.py`
- `labs/lab5/03_train_discrete_denoiser.py`
- `labs/lab5/04_sample_discrete_denoiser.py`
- `labs/lab5/05_make_stage5_figures.py`

### Experiments

Generated:

- `figures/day5/nfe_compare.png`
- `figures/day5/mask_corruption_process.png`
- `figures/day5/discrete_reverse_process.png`
- `figures/day5/final_unified_framework.png`
- `figures/day5/continuous_vs_discrete_diffusion.png`

### Summary

Day 5 connects the continuous experiments to fast sampling and discrete diffusion.

NFE comparison shows that sampling cost is controlled by the number of model evaluations. Fewer evaluations are faster, but generally less accurate.

The toy discrete diffusion lab uses mask corruption:

```text
token -> token or [MASK]
```

This illustrates why categorical data cannot be perturbed by Gaussian noise directly. The reverse model predicts token logits and reconstructs a sequence from a fully masked initial state.

## 7. Day 1-5 Comparison

| Day | Method | Training target | Sampling method |
| --- | --- | --- | --- |
| Day 1 | ODE / SDE simulation | none | numerical simulation |
| Day 2 | Flow Matching | velocity \(z-x_0\) | ODE integration |
| Day 3 | DDPM | noise \(\epsilon\) | reverse denoising chain |
| Day 4 | Conditional DDPM + CFG | noise \(\epsilon\) with class labels | guided reverse denoising |
| Day 5 | Discrete diffusion | token distribution | reverse token denoising |

## 8. Current Status

Day 1-5 code, notes, and figure outputs are complete at the project level.

The Stage 4 figures currently demonstrate the full code path and CFG workflow. For final-quality MNIST samples, run a longer training job:

```bash
conda activate fm_diffusion
python labs/lab4/train_mnist_cfg.py --epochs 20 --batch_size 128 --device cuda
python labs/lab4/eval_cfg_scales.py --scales 0 1 2 4 7 --device cuda
```

For Stage 5 discrete diffusion:

```bash
python labs/lab5/03_train_discrete_denoiser.py --epochs 50 --batch_size 128 --device cuda
python labs/lab5/04_sample_discrete_denoiser.py --device cuda
```
