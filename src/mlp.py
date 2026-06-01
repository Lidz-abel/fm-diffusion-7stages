import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal embedding for scalar time t in [0, 1].

    Input:
        t: shape [batch, 1]
    Output:
        emb: shape [batch, dim]
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even.")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = t.device

        freqs = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                half_dim,
                device=device,
                dtype=t.dtype,
            )
        )

        args = t * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class FlowMLP(nn.Module):
    """
    MLP vector field v_theta(x, t).

    Input:
        x: [batch, 2]
        t: [batch, 1]

    Output:
        velocity: [batch, 2]
    """

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 128,
        time_emb_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)

        input_dim = data_dim + time_emb_dim

        layers = []
        dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.SiLU())
            dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, data_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]

        t_emb = self.time_embedding(t)
        h = torch.cat([x, t_emb], dim=-1)
        return self.net(h)
