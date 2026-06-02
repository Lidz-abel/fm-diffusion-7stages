# Day 5: Unified Theory, Fast Sampling, and Discrete Diffusion

## 1. Goal

Day 5 summarizes the continuous generative modeling tools from Day 1-4 and adds a toy discrete diffusion example.

The key questions are:

1. Why are continuous diffusion models slow?
2. How does NFE control sampling cost?
3. Why does discrete diffusion need transition/corruption instead of Gaussian noise?

## 2. Continuous Models

Flow Matching learns a velocity field:

$$
dX_t=v_\theta(X_t,t)dt.
$$

DDPM learns a noise predictor:

$$
\epsilon_\theta(x_t,t)\approx \epsilon.
$$

Score-SDE learns a score:

$$
s_\theta(x_t,t)\approx \nabla_x\log p_t(x_t).
$$

CFG modifies the denoising direction using conditional and unconditional predictions:

$$
\epsilon_{\mathrm{cfg}}
=
\epsilon_{\mathrm{uncond}}
+s(\epsilon_{\mathrm{cond}}-\epsilon_{\mathrm{uncond}}).
$$

## 3. Fast Sampling

Sampling is expensive when the reverse process requires many function evaluations. NFE measures the number of model calls.

For Flow Matching, reducing NFE means using fewer ODE solver steps. This is faster but may reduce sample quality.

Consistency ideas aim to learn a direct map between noisy and clean states so that fewer steps are needed.

## 4. Discrete Diffusion

Discrete data cannot use Gaussian perturbations directly. Instead, the forward process is a transition or corruption process.

In the toy mask diffusion lab:

```text
token -> token or [MASK]
```

The model learns:

```text
model(x_t, t) -> logits over original tokens
```

Training target:

$$
\mathcal L=\mathrm{CE}(p_\theta(x_0|x_t,t), x_0).
$$

## 5. Continuous vs Discrete

| Aspect | Continuous diffusion | Discrete diffusion |
| --- | --- | --- |
| State | real-valued vectors/images | categorical tokens |
| Forward noise | Gaussian noise | transition / mask corruption |
| Model target | noise, score, or velocity | token distribution |
| Reverse process | denoising vectors | recovering tokens |

## 6. Stage 5 Outputs

- [ ] `figures/day5/nfe_compare.png`
- [ ] `figures/day5/mask_corruption_process.png`
- [ ] `figures/day5/discrete_reverse_process.png`
- [ ] `figures/day5/final_unified_framework.png`
- [ ] `figures/day5/continuous_vs_discrete_diffusion.png`
