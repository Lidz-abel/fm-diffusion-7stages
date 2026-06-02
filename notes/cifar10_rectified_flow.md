# CIFAR-10 Rectified Flow Baseline

Phase 1 moves the project from toy data and MNIST to CIFAR-10 image generation.

The training path is:

$$
x_t = (1-t)x_0 + t x_1,
\qquad x_0 \sim \mathcal N(0,I),
\qquad x_1 \sim p_{\mathrm{data}}.
$$

The target velocity is:

$$
u_t = x_1 - x_0.
$$

The U-Net is trained with MSE regression:

$$
\mathcal L =
\mathbb E \|v_\theta(x_t,t,y) - (x_1-x_0)\|^2.
$$

Sampling solves the ODE:

$$
\frac{dx}{dt}=v_\theta(x_t,t,y),
\qquad t:0\rightarrow1.
$$

The baseline supports CIFAR-10 class conditioning and optional label dropout for CFG-like velocity guidance.
