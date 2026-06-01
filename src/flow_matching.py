from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class FlowMatchingBatch:
    x0: torch.Tensor
    z: torch.Tensor
    t: torch.Tensor
    xt: torch.Tensor
    target_velocity: torch.Tensor


def sample_fm_batch(
    data: torch.Tensor,
    noise_scale: float = 1.0,
) -> FlowMatchingBatch:
    """
    Construct a Flow Matching training batch using straight-line path.

    Path:
        x_t = (1 - t) x_0 + t z

    Target velocity:
        u_t = z - x_0

    Args:
        data: clean data samples z, shape [batch, dim]
        noise_scale: scale of Gaussian initial noise

    Returns:
        FlowMatchingBatch
    """
    z = data
    batch_size, dim = z.shape
    device = z.device

    x0 = noise_scale * torch.randn(batch_size, dim, device=device)
    t = torch.rand(batch_size, 1, device=device)

    xt = (1.0 - t) * x0 + t * z
    target_velocity = z - x0

    return FlowMatchingBatch(
        x0=x0,
        z=z,
        t=t,
        xt=xt,
        target_velocity=target_velocity,
    )


def flow_matching_loss(
    model,
    data: torch.Tensor,
    noise_scale: float = 1.0,
) -> torch.Tensor:
    """
    Flow Matching MSE loss.

    L = E || v_theta(x_t, t) - (z - x_0) ||^2
    """
    batch = sample_fm_batch(data=data, noise_scale=noise_scale)
    pred_velocity = model(batch.xt, batch.t)
    loss = F.mse_loss(pred_velocity, batch.target_velocity)
    return loss
