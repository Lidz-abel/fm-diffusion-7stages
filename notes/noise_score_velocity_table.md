# Noise, Score, and Velocity

| Quantity | Model target | Typical objective | Sampling view |
| --- | --- | --- | --- |
| Noise prediction | \(\epsilon_\theta(x_t,t)\approx\epsilon\) | MSE against injected Gaussian noise | DDPM reverse denoising chain |
| Score prediction | \(s_\theta(x_t,t)\approx\nabla_x\log p_t(x_t)\) | Denoising score matching | Reverse SDE / probability flow ODE |
| Velocity prediction | \(v_\theta(x_t,t)\approx u_t(x_t)\) | Flow Matching regression | ODE transport from prior to data |

## Relationship

For the DDPM forward marginal

$$
x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon,
$$

noise prediction is closely related to score prediction:

$$
\nabla_{x_t}\log q(x_t|x_0)
=
-\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}.
$$

Thus a noise predictor can be converted into a conditional score estimate by scaling.

Flow Matching instead learns a vector field that directly moves samples along a probability path.
