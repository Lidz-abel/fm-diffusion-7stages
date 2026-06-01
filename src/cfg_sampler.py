import torch

from .diffusion import DDPMSchedule, _extract


@torch.no_grad()
def predict_eps_with_cfg(
    model,
    xt: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
    null_label: int,
) -> torch.Tensor:
    """
    Classifier-free guidance:
        eps_cfg = eps_uncond + scale * (eps_cond - eps_uncond)
    """
    if t.ndim == 0:
        t = t.expand(xt.shape[0])
    if y.ndim == 0:
        y = y.expand(xt.shape[0])

    y_null = torch.full_like(y, null_label)
    eps_cond = model(xt, t, y)
    eps_uncond = model(xt, t, y_null)
    return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


@torch.no_grad()
def p_sample_step_cfg(
    model,
    xt: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    schedule: DDPMSchedule,
    cfg_scale: float,
    null_label: int,
) -> torch.Tensor:
    """
    One DDPM reverse step using classifier-free guidance.
    """
    if t.ndim == 0:
        t = t.expand(xt.shape[0])
    if t.ndim != 1:
        t = t.reshape(-1)
    if y.ndim != 1:
        y = y.reshape(-1)

    pred_noise = predict_eps_with_cfg(
        model=model,
        xt=xt,
        t=t,
        y=y,
        cfg_scale=cfg_scale,
        null_label=null_label,
    )

    sqrt_recip_alpha_t = _extract(schedule.sqrt_recip_alphas, t, xt.shape)
    beta_over_sqrt_one_minus_alpha_bar_t = _extract(
        schedule.beta_over_sqrt_one_minus_alpha_bars,
        t,
        xt.shape,
    )
    mean = sqrt_recip_alpha_t * (xt - beta_over_sqrt_one_minus_alpha_bar_t * pred_noise)

    beta_t = _extract(schedule.betas, t, xt.shape)
    noise = torch.randn_like(xt)
    nonzero_mask = (t != 0).float().reshape(xt.shape[0], *([1] * (xt.ndim - 1)))
    return mean + nonzero_mask * torch.sqrt(beta_t) * noise


@torch.no_grad()
def sample_ddpm_cfg(
    model,
    shape: tuple[int, int, int, int],
    y: torch.Tensor,
    schedule: DDPMSchedule,
    cfg_scale: float,
    null_label: int,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Full DDPM sampling loop with classifier-free guidance.
    """
    x = torch.randn(shape, device=device)
    y = y.to(device)
    if y.shape[0] != shape[0]:
        raise ValueError(f"y batch size {y.shape[0]} does not match sample shape {shape}.")

    for step in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        x = p_sample_step_cfg(
            model=model,
            xt=x,
            t=t,
            y=y,
            schedule=schedule,
            cfg_scale=cfg_scale,
            null_label=null_label,
        )
    return x
