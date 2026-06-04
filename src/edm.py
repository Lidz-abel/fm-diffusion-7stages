from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class EDMLossConfig:
    sigma_data: float = 0.5
    p_mean: float = -1.2
    p_std: float = 1.2


@dataclass
class EDMSamplerConfig:
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    num_steps: int = 40


def apply_label_dropout(
    y: torch.Tensor,
    null_label: int,
    drop_label_prob: float,
) -> torch.Tensor:
    if drop_label_prob <= 0:
        return y
    y_train = y.clone()
    drop_mask = torch.rand(y.shape[0], device=y.device) < drop_label_prob
    y_train[drop_mask] = null_label
    return y_train


def sample_log_normal_sigmas(
    batch_size: int,
    device: torch.device,
    p_mean: float = -1.2,
    p_std: float = 1.2,
) -> torch.Tensor:
    return torch.randn(batch_size, device=device).mul(p_std).add(p_mean).exp()


def edm_scalings(
    sigma: torch.Tensor,
    sigma_data: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sigma_data_tensor = torch.as_tensor(sigma_data, device=sigma.device, dtype=sigma.dtype)
    sigma2 = sigma.square()
    sigma_data2 = sigma_data_tensor.square()
    c_skip = sigma_data2 / (sigma2 + sigma_data2)
    c_out = sigma * sigma_data_tensor / (sigma2 + sigma_data2).sqrt()
    c_in = 1.0 / (sigma2 + sigma_data2).sqrt()
    c_noise = sigma.log() / 4.0
    return c_skip, c_out, c_in, c_noise


def edm_denoise(
    model,
    x: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    sigma_data: float = 0.5,
    cfg_scale: float = 1.0,
    null_label: int | None = None,
) -> torch.Tensor:
    if sigma.ndim == 0:
        sigma = sigma.expand(x.shape[0])
    sigma = sigma.to(device=x.device, dtype=x.dtype)
    c_skip, c_out, c_in, c_noise = edm_scalings(sigma=sigma, sigma_data=sigma_data)
    model_input = c_in[:, None, None, None] * x

    if cfg_scale == 1.0 or null_label is None:
        model_out = model(model_input, c_noise, y)
    else:
        y_null = torch.full_like(y, fill_value=null_label)
        model_out_cond = model(model_input, c_noise, y)
        model_out_uncond = model(model_input, c_noise, y_null)
        model_out = model_out_uncond + cfg_scale * (model_out_cond - model_out_uncond)

    return c_skip[:, None, None, None] * x + c_out[:, None, None, None] * model_out


def edm_loss(
    model,
    x0: torch.Tensor,
    y: torch.Tensor,
    null_label: int,
    drop_label_prob: float = 0.1,
    config: EDMLossConfig | None = None,
) -> torch.Tensor:
    config = config or EDMLossConfig()
    sigmas = sample_log_normal_sigmas(
        batch_size=x0.shape[0],
        device=x0.device,
        p_mean=config.p_mean,
        p_std=config.p_std,
    )
    noise = torch.randn_like(x0)
    x_noisy = x0 + sigmas[:, None, None, None] * noise
    y_train = apply_label_dropout(y, null_label=null_label, drop_label_prob=drop_label_prob)
    denoised = edm_denoise(
        model=model,
        x=x_noisy,
        sigma=sigmas,
        y=y_train,
        sigma_data=config.sigma_data,
    )
    weight = (sigmas.square() + config.sigma_data**2) / (sigmas * config.sigma_data).square()
    loss = weight[:, None, None, None] * (denoised - x0).square()
    return loss.mean()


def karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    device: torch.device,
) -> torch.Tensor:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, torch.zeros(1, device=device)])


@torch.no_grad()
def sample_edm(
    model,
    shape: tuple[int, int, int, int],
    y: torch.Tensor,
    device: torch.device,
    config: EDMSamplerConfig | None = None,
    cfg_scale: float = 1.0,
    null_label: int | None = None,
    solver: str = "heun",
) -> torch.Tensor:
    config = config or EDMSamplerConfig()
    if solver not in {"euler", "heun"}:
        raise ValueError("solver must be 'euler' or 'heun'.")

    sigmas = karras_sigmas(
        num_steps=config.num_steps,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        rho=config.rho,
        device=device,
    )
    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(config.num_steps):
        sigma = sigmas[i].expand(shape[0])
        sigma_next = sigmas[i + 1].expand(shape[0])
        denoised = edm_denoise(
            model=model,
            x=x,
            sigma=sigma,
            y=y,
            sigma_data=config.sigma_data,
            cfg_scale=cfg_scale,
            null_label=null_label,
        )
        d = (x - denoised) / sigma[:, None, None, None]
        x_euler = x + (sigma_next - sigma)[:, None, None, None] * d

        if solver == "heun" and sigmas[i + 1] > 0:
            denoised_next = edm_denoise(
                model=model,
                x=x_euler,
                sigma=sigma_next,
                y=y,
                sigma_data=config.sigma_data,
                cfg_scale=cfg_scale,
                null_label=null_label,
            )
            d_next = (x_euler - denoised_next) / sigma_next[:, None, None, None]
            x = x + (sigma_next - sigma)[:, None, None, None] * (0.5 * d + 0.5 * d_next)
        else:
            x = x_euler

    return x.clamp(-1.0, 1.0)
