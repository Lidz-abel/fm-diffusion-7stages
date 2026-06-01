from dataclasses import dataclass

import torch
import torch.nn.functional as F


def _extract(values: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Gather per-timestep coefficients and reshape them for broadcasting to x.
    """
    if t.ndim != 1:
        t = t.reshape(-1)
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(x_shape) - 1)))


@dataclass
class DDPMSchedule:
    """
    Diffusion schedule for DDPM.

    Timesteps are indexed from 0 to T - 1.
    """

    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    device: str | torch.device = "cpu"

    def __post_init__(self) -> None:
        if self.timesteps <= 0:
            raise ValueError("timesteps must be positive.")
        if not 0.0 < self.beta_start < self.beta_end < 1.0:
            raise ValueError("expected 0 < beta_start < beta_end < 1.")

        self.betas = torch.linspace(
            self.beta_start,
            self.beta_end,
            self.timesteps,
            device=self.device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.beta_over_sqrt_one_minus_alpha_bars = self.betas / self.sqrt_one_minus_alpha_bars

    def to(self, device: str | torch.device) -> "DDPMSchedule":
        self.device = device
        for name in [
            "betas",
            "alphas",
            "alpha_bars",
            "sqrt_alpha_bars",
            "sqrt_one_minus_alpha_bars",
            "sqrt_recip_alphas",
            "beta_over_sqrt_one_minus_alpha_bars",
        ]:
            setattr(self, name, getattr(self, name).to(device))
        return self


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    """
    Forward noising:
        x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) noise
    """
    if x0.shape != noise.shape:
        raise ValueError(f"x0 shape {x0.shape} must match noise shape {noise.shape}.")

    sqrt_alpha_bar_t = _extract(schedule.sqrt_alpha_bars, t, x0.shape)
    sqrt_one_minus_alpha_bar_t = _extract(schedule.sqrt_one_minus_alpha_bars, t, x0.shape)
    return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise


def ddpm_loss(
    model,
    x0: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    """
    DDPM noise prediction loss:
        L = E || epsilon_theta(x_t, t) - epsilon ||^2
    """
    batch_size = x0.shape[0]
    t = torch.randint(0, schedule.timesteps, (batch_size,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = q_sample(x0=x0, t=t, noise=noise, schedule=schedule)

    t_model = t.float()[:, None] / max(schedule.timesteps - 1, 1)
    pred_noise = model(xt, t_model)
    return F.mse_loss(pred_noise, noise)


@torch.no_grad()
def p_sample_step(
    model,
    xt: torch.Tensor,
    t: torch.Tensor,
    schedule: DDPMSchedule,
) -> torch.Tensor:
    """
    One reverse denoising step: x_t -> x_{t-1}.
    """
    if t.ndim == 0:
        t = t.expand(xt.shape[0])
    if t.ndim != 1:
        t = t.reshape(-1)
    if t.shape[0] != xt.shape[0]:
        raise ValueError(f"t shape {t.shape} is incompatible with xt shape {xt.shape}.")

    t_model = t.float()[:, None] / max(schedule.timesteps - 1, 1)
    pred_noise = model(xt, t_model)

    sqrt_recip_alpha_t = _extract(schedule.sqrt_recip_alphas, t, xt.shape)
    beta_over_sqrt_one_minus_alpha_bar_t = _extract(
        schedule.beta_over_sqrt_one_minus_alpha_bars,
        t,
        xt.shape,
    )

    mean = sqrt_recip_alpha_t * (xt - beta_over_sqrt_one_minus_alpha_bar_t * pred_noise)

    noise = torch.randn_like(xt)
    beta_t = _extract(schedule.betas, t, xt.shape)
    nonzero_mask = (t != 0).float().reshape(xt.shape[0], *([1] * (xt.ndim - 1)))
    return mean + nonzero_mask * torch.sqrt(beta_t) * noise


@torch.no_grad()
def sample_ddpm(
    model,
    n_samples: int,
    schedule: DDPMSchedule,
    device: str | torch.device = "cpu",
    dim: int = 2,
) -> torch.Tensor:
    """
    Start from Gaussian noise and run reverse denoising.
    """
    x = torch.randn(n_samples, dim, device=device)
    for step in reversed(range(schedule.timesteps)):
        t = torch.full((n_samples,), step, device=device, dtype=torch.long)
        x = p_sample_step(model=model, xt=x, t=t, schedule=schedule)
    return x


@torch.no_grad()
def sample_ddpm_trajectory(
    model,
    n_samples: int,
    schedule: DDPMSchedule,
    device: str | torch.device = "cpu",
    dim: int = 2,
    save_every: int | None = None,
) -> dict[int, torch.Tensor]:
    """
    Store intermediate denoising states.

    Returns:
        Dictionary mapping reverse timestep index to samples.
    """
    if save_every is None:
        save_every = max(schedule.timesteps // 4, 1)
    if save_every <= 0:
        raise ValueError("save_every must be positive.")

    x = torch.randn(n_samples, dim, device=device)
    trajectory: dict[int, torch.Tensor] = {schedule.timesteps: x.clone()}

    for step in reversed(range(schedule.timesteps)):
        t = torch.full((n_samples,), step, device=device, dtype=torch.long)
        x = p_sample_step(model=model, xt=x, t=t, schedule=schedule)
        if step % save_every == 0 or step == 0:
            trajectory[step] = x.clone()

    return trajectory
