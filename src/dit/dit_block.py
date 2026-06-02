import torch
import torch.nn as nn


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        inner_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(inner_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """
    AdaLN-Zero DiT block.
    """

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_dim, mlp_ratio=mlp_ratio)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN(c).chunk(6, dim=1)

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa[:, None, :] * h

        h = modulate(self.norm2(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        x = x + gate_mlp[:, None, :] * h
        return x
