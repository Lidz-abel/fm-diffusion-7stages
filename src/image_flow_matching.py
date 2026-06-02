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


def sample_rectified_flow_tuple(x1: torch.Tensor) -> RectifiedFlowBatch:
    """
    Build the straight-path Rectified Flow tuple.

    Path:
        x_t = (1 - t) x_0 + t x_1
    Target velocity:
        u_t = x_1 - x_0
    """
    batch_size = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype)
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
) -> torch.Tensor:
    """
    Class-conditional image Flow Matching loss.

    L = E || v_theta(x_t, t, y) - (x_1 - x_0) ||^2
    """
    batch = sample_rectified_flow_tuple(x1)
    y_train = apply_label_dropout(y.to(x1.device), null_label, drop_label_prob)
    pred_v = model(batch.xt, batch.t, y_train)
    return F.mse_loss(pred_v, batch.target_v)
