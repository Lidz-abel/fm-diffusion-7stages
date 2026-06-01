from typing import Callable

import math
import torch


def simulate_brownian_motion(
    n_paths: int,
    n_steps: int,
    dim: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Simulate Brownian motion paths.

    Brownian update:
        W_{t+h} = W_t + sqrt(h) * eps_t,
        eps_t ~ N(0, I)

    Args:
        n_paths: number of trajectories
        n_steps: number of time steps
        dim: dimension of Brownian motion
        device: torch device

    Returns:
        paths: Tensor with shape [n_paths, n_steps + 1, dim]
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    h = 1.0 / n_steps
    eps = torch.randn(n_paths, n_steps, dim, device=device)
    increments = math.sqrt(h) * eps

    w0 = torch.zeros(n_paths, 1, dim, device=device)
    paths = torch.cat([w0, increments.cumsum(dim=1)], dim=1)
    return paths


@torch.no_grad()
def euler_maruyama_solver(
    x0: torch.Tensor,
    drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    sigma: Callable[[torch.Tensor], torch.Tensor | float],
    n_steps: int,
) -> torch.Tensor:
    """
    Solve SDE:
        dX_t = u_t(X_t) dt + sigma_t dW_t

    Euler-Maruyama update:
        X_{t+h} = X_t + h u_t(X_t) + sigma_t sqrt(h) eps_t

    Args:
        x0: Tensor with shape [batch, dim]
        drift: function f(x, t) -> drift, same shape as x
        sigma: function g(t) -> scalar or tensor broadcastable to x
        n_steps: number of time steps

    Returns:
        traj: Tensor with shape [n_steps + 1, batch, dim]
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    h = 1.0 / n_steps
    x = x0.clone()
    traj = [x.clone()]

    for i in range(n_steps):
        t_value = i * h
        t = torch.full((x.shape[0], 1), t_value, device=x.device, dtype=x.dtype)

        drift_value = drift(x, t)
        if drift_value.shape != x.shape:
            raise ValueError(f"drift output shape {drift_value.shape} does not match x shape {x.shape}.")

        noise = torch.randn_like(x)
        sigma_value = sigma(t)

        x = x + h * drift_value + math.sqrt(h) * sigma_value * noise
        traj.append(x.clone())

    return torch.stack(traj, dim=0)