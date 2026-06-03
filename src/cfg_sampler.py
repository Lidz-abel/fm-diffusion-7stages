import torch

from .diffusion import DDPMSchedule, _extract, model_output_to_epsilon


@torch.no_grad()
def predict_eps_with_cfg(
    model,
    xt: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
    null_label: int,
    schedule: DDPMSchedule | None = None,
    prediction_type: str = "epsilon",
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
    pred_cond = model(xt, t, y)
    pred_uncond = model(xt, t, y_null)
    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    if prediction_type == "epsilon":
        return pred
    if schedule is None:
        raise ValueError("schedule is required when prediction_type is not epsilon.")
    return model_output_to_epsilon(
        model_output=pred,
        xt=xt,
        t=t,
        schedule=schedule,
        prediction_type=prediction_type,
    )


@torch.no_grad()
def p_sample_step_cfg(
    model,
    xt: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    schedule: DDPMSchedule,
    cfg_scale: float,
    null_label: int,
    prediction_type: str = "epsilon",
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
        schedule=schedule,
        prediction_type=prediction_type,
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
    prediction_type: str = "epsilon",
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
            prediction_type=prediction_type,
        )
    return x


@torch.no_grad()
def sample_ddim_cfg(
    model,
    shape: tuple[int, int, int, int],
    y: torch.Tensor,
    schedule: DDPMSchedule,
    cfg_scale: float,
    null_label: int,
    prediction_type: str = "epsilon",
    num_steps: int = 100,
    eta: float = 0.0,
    clip_x0: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    DDIM sampling loop with classifier-free guidance.

    eta=0.0 gives deterministic DDIM sampling. Larger eta adds stochasticity.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if num_steps > schedule.timesteps:
        raise ValueError("num_steps cannot exceed schedule.timesteps.")
    if eta < 0.0:
        raise ValueError("eta must be non-negative.")

    x = torch.randn(shape, device=device)
    y = y.to(device)
    if y.shape[0] != shape[0]:
        raise ValueError(f"y batch size {y.shape[0]} does not match sample shape {shape}.")

    steps = torch.linspace(0, schedule.timesteps - 1, num_steps, device=device).round().long()
    steps = torch.unique_consecutive(steps)
    if steps[-1].item() != schedule.timesteps - 1:
        steps = torch.cat([steps, torch.tensor([schedule.timesteps - 1], device=device)])

    for idx in reversed(range(len(steps))):
        step = int(steps[idx].item())
        prev_step = int(steps[idx - 1].item()) if idx > 0 else -1

        t = torch.full((shape[0],), step, device=device, dtype=torch.long)
        pred_noise = predict_eps_with_cfg(
            model=model,
            xt=x,
            t=t,
            y=y,
            cfg_scale=cfg_scale,
            null_label=null_label,
            schedule=schedule,
            prediction_type=prediction_type,
        )

        alpha_bar_t = schedule.alpha_bars[step].reshape(1, 1, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
        if clip_x0:
            pred_x0 = pred_x0.clamp(-1.0, 1.0)

        if prev_step < 0:
            x = pred_x0
            continue

        alpha_bar_prev = schedule.alpha_bars[prev_step].reshape(1, 1, 1, 1)
        sigma = (
            eta
            * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            * torch.sqrt(torch.clamp(1.0 - alpha_bar_t / alpha_bar_prev, min=0.0))
        )
        direction_scale = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0))
        noise = torch.randn_like(x) if eta > 0.0 else torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_prev) * pred_x0 + direction_scale * pred_noise + sigma * noise

    return x
