import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding followed by an MLP for integer diffusion timesteps.

    Input:
        t: [batch] or [batch, 1]
    Output:
        embedding: [batch, dim]
    """

    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even.")

        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError(f"Expected t with shape [batch] or [batch, 1], got {t.shape}.")

        half_dim = self.dim // 2
        device = t.device
        dtype = torch.float32 if not torch.is_floating_point(t) else t.dtype
        t = t.to(dtype)

        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=dtype)
            / max(half_dim - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)
