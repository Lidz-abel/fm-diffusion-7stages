# Course Summary: Flow Matching and Diffusion Models

## 1. Flow Matching

Flow Matching trains a velocity field by supervised regression on probability paths.

For a straight path:

$$
x_t=(1-t)x_0+tz,
$$

the target velocity is:

$$
u_t=z-x_0.
$$

Sampling solves an ODE from a simple prior to the data distribution.

## 2. DDPM

DDPM defines a Gaussian forward noising process:

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon.
$$

The model predicts noise:

$$
\epsilon_\theta(x_t,t)\approx \epsilon.
$$

Sampling starts from Gaussian noise and iteratively denoises.

## 3. Score-SDE

Score-based models learn:

$$
\nabla_x\log p_t(x).
$$

The score can define a reverse SDE or a probability flow ODE. Noise prediction and score prediction are closely related under Gaussian perturbations.

## 4. Classifier-Free Guidance

CFG trains one model with conditional and unconditional examples. At sampling time:

$$
\epsilon_{\mathrm{cfg}}
=
\epsilon_{\mathrm{uncond}}
+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
$$

Higher guidance scale usually improves condition consistency but can reduce diversity.

## 5. Latent Diffusion

Latent diffusion performs denoising in an encoded latent space instead of pixel space:

```text
image -> encoder -> latent diffusion -> decoder -> image
```

This reduces computational cost.

## 6. DiT

DiT replaces the U-Net denoiser with a Transformer over patch tokens.

```text
x_t image -> patchify -> DiT blocks -> predicted noise patches -> unpatchify
```

The diffusion objective is unchanged. DiT is an architecture choice for the denoising network, not a new training target.

## 7. Discrete Diffusion

Discrete diffusion replaces Gaussian noise with categorical transitions.

The Stage 5 toy lab uses mask corruption:

```text
token -> token or [MASK]
```

The reverse model predicts original tokens from corrupted sequences.

## 8. Unified View

| Method | State space | Target | Sampling |
| --- | --- | --- | --- |
| Flow Matching | continuous | velocity | ODE |
| DDPM | continuous | noise | reverse denoising |
| Score-SDE | continuous | score | reverse SDE / ODE |
| CFG | continuous or latent | guided denoising | conditional reverse process |
| DiT | continuous or latent | architecture for prediction | patch-token denoising |
| Discrete diffusion | categorical | token distribution | reverse token denoising |
