# Stage 3: Score Matching and DDPM

## 1. Goal

Stage 3 implements a minimal 2D DDPM on toy data. The model is trained as a time-conditioned noise predictor.

The forward process adds Gaussian noise to data:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

The model learns:

$$
\epsilon_\theta(x_t,t)\approx \epsilon.
$$

## 2. DDPM Schedule

For timesteps \(t=0,\ldots,T-1\), define:

$$
\beta_t,\qquad
\alpha_t=1-\beta_t,\qquad
\bar\alpha_t=\prod_{s=0}^{t}\alpha_s.
$$

The implementation stores:

- `betas`
- `alphas`
- `alpha_bars`
- `sqrt_alpha_bars`
- `sqrt_one_minus_alpha_bars`
- `sqrt_recip_alphas`
- `beta_over_sqrt_one_minus_alpha_bars`

## 3. Forward Noising

The function `q_sample` samples \(x_t\) in one step:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

This works because the Gaussian forward Markov chain has a closed-form marginal distribution \(q(x_t|x_0)\).

## 4. Training Objective

The loss is:

$$
\mathcal L_{\mathrm{DDPM}}
=
\mathbb E
\left[
\|\epsilon_\theta(x_t,t)-\epsilon\|^2
\right].
$$

The model takes both \(x_t\) and \(t\) because the amount of noise depends on the timestep.

## 5. Reverse Sampling

Sampling starts from:

$$
x_T\sim\mathcal N(0,I).
$$

Then DDPM applies reverse denoising steps:

$$
x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0.
$$

The reverse mean uses:

$$
\mu_\theta(x_t,t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\epsilon_\theta(x_t,t)
\right).
$$

At \(t=0\), no extra random noise is added.

## 6. DDPM vs Flow Matching

Flow Matching trains a velocity field for an ODE:

$$
dX_t=v_\theta(X_t,t)dt.
$$

DDPM trains a noise predictor for an iterative denoising process:

$$
\epsilon_\theta(x_t,t)\approx\epsilon.
$$

Both start sampling from Gaussian noise, but Flow Matching integrates an ODE from \(t=0\) to \(t=1\), while DDPM iterates a reverse denoising chain from \(T\) to \(0\).

## 7. Stage 3 Outputs

- [ ] `src/diffusion.py`
- [ ] `labs/lab2/run_2d_ddpm.py`
- [ ] `figures/stage3/forward_noising.png`
- [ ] `figures/stage3/ddpm_samples.png`
- [ ] `figures/stage3/ddpm_sampling_trajectory.png`
- [ ] `figures/stage3/fm_vs_ddpm.png`
- [ ] `checkpoints/stage3_ddpm.pt`

## 8. Check Questions

1. Why can `q_sample` obtain \(x_t\) in one step?
2. Why is the training target noise \(\epsilon\)?
3. Why does the model need both \(x_t\) and \(t\)?
4. Why does DDPM sampling start from Gaussian noise?
5. Why is no random noise added at \(t=0\)?
6. How is DDPM training different from Flow Matching training?
7. How is DDPM sampling different from Flow Matching sampling?
