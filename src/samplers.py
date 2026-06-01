import torch


@torch.no_grad()
def sample_ode_euler(
    model,
    x0: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """
    Euler ODE sampler for learned vector field.

    dX_t = v_theta(X_t, t) dt

    Args:
        model: vector field model
        x0: initial noise, shape [batch, dim]
        n_steps: number of function evaluations

    Returns:
        x: final sample, shape [batch, dim]
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    h = 1.0 / n_steps
    x = x0.clone()

    for i in range(n_steps):
        t_value = i * h
        t = torch.full((x.shape[0], 1), t_value, device=x.device, dtype=x.dtype)
        v = model(x, t)
        x = x + h * v

    return x


@torch.no_grad()
def sample_ode_euler_trajectory(
    model,
    x0: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """
    Euler ODE sampler that stores the whole trajectory.

    Returns:
        traj: shape [n_steps + 1, batch, dim]
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    h = 1.0 / n_steps
    x = x0.clone()
    traj = [x.clone()]

    for i in range(n_steps):
        t_value = i * h
        t = torch.full((x.shape[0], 1), t_value, device=x.device, dtype=x.dtype)
        v = model(x, t)
        x = x + h * v
        traj.append(x.clone())

    return torch.stack(traj, dim=0)


@torch.no_grad()
def sample_ode_heun(
    model,
    x0: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """
    Heun ODE sampler.

    Predictor:
        x' = x + h v_theta(x, t)

    Corrector:
        x_next = x + h/2 * [v_theta(x,t) + v_theta(x', t+h)]
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")

    h = 1.0 / n_steps
    x = x0.clone()

    for i in range(n_steps):
        t_value = i * h
        t = torch.full((x.shape[0], 1), t_value, device=x.device, dtype=x.dtype)

        v1 = model(x, t)
        x_pred = x + h * v1

        t_next_value = min((i + 1) * h, 1.0)
        t_next = torch.full((x.shape[0], 1), t_next_value, device=x.device, dtype=x.dtype)

        v2 = model(x_pred, t_next)
        x = x + 0.5 * h * (v1 + v2)

    return x
