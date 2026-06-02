import torch


@torch.no_grad()
def predict_velocity_with_cfg(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float = 1.0,
    null_label: int = 10,
) -> torch.Tensor:
    """
    CFG-style velocity prediction:
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
    """
    if cfg_scale == 1.0:
        return model(x, t, y)

    y_null = torch.full_like(y, null_label)
    v_uncond = model(x, t, y_null)
    if cfg_scale == 0.0:
        return v_uncond
    v_cond = model(x, t, y)
    return v_uncond + cfg_scale * (v_cond - v_uncond)


@torch.no_grad()
def sample_euler(
    model,
    shape: tuple[int, ...],
    y: torch.Tensor,
    nfe: int,
    device: str | torch.device,
    cfg_scale: float = 1.0,
    null_label: int = 10,
) -> torch.Tensor:
    """
    Euler ODE sampler for Rectified Flow.
    """
    if nfe <= 0:
        raise ValueError("nfe must be positive.")

    device = torch.device(device)
    x = torch.randn(shape, device=device)
    y = y.to(device)
    dt = 1.0 / nfe

    for step in range(nfe):
        t = torch.full((shape[0],), step * dt, device=device, dtype=x.dtype)
        v = predict_velocity_with_cfg(model, x, t, y, cfg_scale, null_label)
        x = x + dt * v
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def sample_heun(
    model,
    shape: tuple[int, ...],
    y: torch.Tensor,
    nfe: int,
    device: str | torch.device,
    cfg_scale: float = 1.0,
    null_label: int = 10,
) -> torch.Tensor:
    """
    Heun predictor-corrector ODE sampler for Rectified Flow.
    """
    if nfe <= 0:
        raise ValueError("nfe must be positive.")

    device = torch.device(device)
    x = torch.randn(shape, device=device)
    y = y.to(device)
    dt = 1.0 / nfe

    for step in range(nfe):
        t_value = step * dt
        t = torch.full((shape[0],), t_value, device=device, dtype=x.dtype)
        v1 = predict_velocity_with_cfg(model, x, t, y, cfg_scale, null_label)
        x_pred = x + dt * v1
        t_next = torch.full(
            (shape[0],),
            min(t_value + dt, 1.0),
            device=device,
            dtype=x.dtype,
        )
        v2 = predict_velocity_with_cfg(model, x_pred, t_next, y, cfg_scale, null_label)
        x = x + 0.5 * dt * (v1 + v2)
    return x.clamp(-1.0, 1.0)


@torch.no_grad()
def sample_flow(
    model,
    shape: tuple[int, ...],
    y: torch.Tensor,
    solver: str,
    nfe: int,
    device: str | torch.device,
    cfg_scale: float = 1.0,
    null_label: int = 10,
) -> torch.Tensor:
    """
    Unified Rectified Flow sampler.
    """
    solver = solver.lower()
    if solver == "euler":
        return sample_euler(model, shape, y, nfe, device, cfg_scale, null_label)
    if solver == "heun":
        return sample_heun(model, shape, y, nfe, device, cfg_scale, null_label)
    raise ValueError(f"Unknown solver: {solver}. Expected 'euler' or 'heun'.")
