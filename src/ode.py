from typing import Callable

import torch


@torch.no_grad()
def euler_solver(
    x0: torch.Tensor,
    vector_field: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    n_steps: int,
) -> torch.Tensor:
    """
    Solve ODE:
        dX_t = u_t(X_t) dt

    Euler update:
        X_{t+h} = X_t + h u_t(X_t)

    Args:
        x0: Tensor with shape [batch, dim]
        vector_field: function f(x, t) -> velocity, same shape as x
        n_steps: number of Euler steps

    Returns:
        traj: Tensor with shape [n_steps + 1, batch, dim]
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    h = 1.0 / n_steps
    x = x0.clone()
    traj = [x.clone()]

    for i in range(n_steps):
        t = torch.full((x.shape[0], 1), i * h, device=x.device, dtype=x.dtype)
        v = vector_field(x, t)

        if v.shape != x.shape:
            raise ValueError(f"vector_field output shape {v.shape} does not match x shape {x.shape}.")

        x = x + h * v
        traj.append(x.clone())

    return torch.stack(traj, dim=0)