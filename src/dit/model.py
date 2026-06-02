import torch
import torch.nn as nn

from .dit_block import DiTBlock, modulate
from .embeddings import LabelEmbedder, TimestepEmbedder
from .patch_embed import PatchEmbed, unpatchify


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, patch_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.linear = nn.Linear(hidden_dim, patch_dim)
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


class MiniDiT(nn.Module):
    """
    Small DiT denoiser for MNIST diffusion.
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_classes: int = 10,
        null_label: int = 10,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.null_label = null_label

        self.patch_embed = PatchEmbed(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, hidden_dim))
        self.time_embed = TimestepEmbedder(hidden_dim=hidden_dim, frequency_dim=hidden_dim)
        self.label_embed = LabelEmbedder(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            null_label=null_label,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, self.patch_embed.patch_dim)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        drop_label_prob: float = 0.0,
    ) -> torch.Tensor:
        tokens = self.patch_embed(x) + self.pos_embed
        c = self.time_embed(t) + self.label_embed(
            y,
            train=self.training,
            drop_label_prob=drop_label_prob,
        )

        for block in self.blocks:
            tokens = block(tokens, c)

        pred_patches = self.final_layer(tokens, c)
        return unpatchify(
            pred_patches,
            patch_size=self.patch_size,
            image_size=self.image_size,
            channels=self.in_channels,
        )

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        y_null = torch.full_like(y, self.null_label)
        eps_cond = self.forward(x, t, y, drop_label_prob=0.0)
        eps_uncond = self.forward(x, t, y_null, drop_label_prob=0.0)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)
