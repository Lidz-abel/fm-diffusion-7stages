from __future__ import annotations

import torch


def tensor_summary(images: torch.Tensor) -> dict[str, float]:
    """
    Lightweight image summary for experiment CSVs.
    """
    x = images.detach().float().cpu()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }
