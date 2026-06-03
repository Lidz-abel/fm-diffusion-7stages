from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class RectifiedFlowBatch:
    x0: torch.Tensor
    x1: torch.Tensor
    t: torch.Tensor
    xt: torch.Tensor
    target_v: torch.Tensor


def sample_time(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    time_sampling: str = "uniform",
    time_beta_alpha: float = 1.0,
    time_beta_beta: float = 1.0,
) -> torch.Tensor:
    """
    Sample continuous Rectified Flow time values in [0, 1].
    """
    if time_sampling == "uniform":
        return torch.rand(batch_size, device=device, dtype=dtype)
    if time_sampling == "beta":
        if time_beta_alpha <= 0.0 or time_beta_beta <= 0.0:
            raise ValueError("Beta time sampling requires positive alpha and beta.")
        alpha = torch.tensor(time_beta_alpha, device=device, dtype=dtype)
        beta = torch.tensor(time_beta_beta, device=device, dtype=dtype)
        return torch.distributions.Beta(alpha, beta).sample((batch_size,))
    raise ValueError(f"Unknown time_sampling: {time_sampling}.")


def loss_weight_from_time(
    t: torch.Tensor,
    loss_weighting: str = "none",
    loss_weight_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Per-sample loss weighting for image Rectified Flow.

    - none: all samples weighted equally.
    - data_end: emphasize late times near the data endpoint.
    - middle: emphasize the middle of the path.
    """
    if loss_weighting == "none":
        return torch.ones_like(t)
    if loss_weighting == "data_end":
        return 1.0 + loss_weight_lambda * t
    if loss_weighting == "middle":
        return 1.0 + loss_weight_lambda * (4.0 * t * (1.0 - t))
    raise ValueError(f"Unknown loss_weighting: {loss_weighting}.")


def apply_label_dropout(
    y: torch.Tensor,
    null_label: int | None,
    drop_label_prob: float,
) -> torch.Tensor:
    """
    Replace a random subset of labels with null_label for CFG-style training.
    """
    if not 0.0 <= drop_label_prob <= 1.0:
        raise ValueError("drop_label_prob must be in [0, 1].")
    if null_label is None or drop_label_prob == 0.0:
        return y

    y_train = y.clone()
    drop_mask = torch.rand(y.shape[0], device=y.device) < drop_label_prob
    y_train[drop_mask] = null_label
    return y_train


def sample_rectified_flow_tuple(
    x1: torch.Tensor,
    time_sampling: str = "uniform",
    time_beta_alpha: float = 1.0,
    time_beta_beta: float = 1.0,
) -> RectifiedFlowBatch:
    """
    Build the straight-path Rectified Flow tuple.

    Path:
        x_t = (1 - t) x_0 + t x_1
    Target velocity:
        u_t = x_1 - x_0
    """
    batch_size = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = sample_time(
        batch_size=batch_size,
        device=x1.device,
        dtype=x1.dtype,
        time_sampling=time_sampling,
        time_beta_alpha=time_beta_alpha,
        time_beta_beta=time_beta_beta,
    )
    t_view = t.reshape(batch_size, *([1] * (x1.ndim - 1)))
    xt = (1.0 - t_view) * x0 + t_view * x1
    target_v = x1 - x0
    return RectifiedFlowBatch(x0=x0, x1=x1, t=t, xt=xt, target_v=target_v)


def flow_matching_loss(
    model,
    x1: torch.Tensor,
    y: torch.Tensor,
    null_label: int | None = None,
    drop_label_prob: float = 0.0,
    time_sampling: str = "uniform",
    time_beta_alpha: float = 1.0,
    time_beta_beta: float = 1.0,
    loss_weighting: str = "none",
    loss_weight_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Class-conditional image Flow Matching loss.

    L = E || v_theta(x_t, t, y) - (x_1 - x_0) ||^2
    """
    batch = sample_rectified_flow_tuple(
        x1,
        time_sampling=time_sampling,
        time_beta_alpha=time_beta_alpha,
        time_beta_beta=time_beta_beta,
    )
    y_train = apply_label_dropout(y.to(x1.device), null_label, drop_label_prob)
    pred_v = model(batch.xt, batch.t, y_train)
    per_sample_loss = F.mse_loss(pred_v, batch.target_v, reduction="none")
    per_sample_loss = per_sample_loss.reshape(x1.shape[0], -1).mean(dim=1)
    weights = loss_weight_from_time(
        batch.t,
        loss_weighting=loss_weighting,
        loss_weight_lambda=loss_weight_lambda,
    )
    return (weights * per_sample_loss).mean()
