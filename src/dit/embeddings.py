import math

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """
    t -> sinusoidal embedding -> MLP -> hidden_dim.
    """

    def __init__(self, hidden_dim: int, frequency_dim: int = 256, max_period: int = 10_000):
        super().__init__()
        if frequency_dim % 2 != 0:
            raise ValueError("frequency_dim must be even.")
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [B], got {t.shape}.")

        half = self.frequency_dim // 2
        dtype = torch.float32 if not torch.is_floating_point(t) else t.dtype
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=dtype)
            / max(half - 1, 1)
        )
        args = t.to(dtype)[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class LabelEmbedder(nn.Module):
    """
    Label embedding with a null label for classifier-free guidance.
    """

    def __init__(self, num_classes: int = 10, hidden_dim: int = 192, null_label: int = 10):
        super().__init__()
        if null_label < num_classes:
            raise ValueError("null_label must be outside the normal class range.")
        self.num_classes = num_classes
        self.null_label = null_label
        self.embedding = nn.Embedding(null_label + 1, hidden_dim)

    def forward(
        self,
        y: torch.Tensor,
        train: bool = False,
        drop_label_prob: float = 0.0,
    ) -> torch.Tensor:
        if y.ndim != 1:
            y = y.reshape(-1)
        y = y.clone()
        if train and drop_label_prob > 0.0:
            drop_mask = torch.rand(y.shape[0], device=y.device) < drop_label_prob
            y[drop_mask] = self.null_label
        y = y.clamp(min=0, max=self.null_label)
        return self.embedding(y)
