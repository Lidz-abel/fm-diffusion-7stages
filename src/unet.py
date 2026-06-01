import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embedding import SinusoidalTimeEmbedding


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.norm1 = nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=_num_groups(out_channels), num_channels=out_channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        return h + self.skip(x)


class MNISTConditionalUNet(nn.Module):
    """
    Lightweight class-conditional U-Net for MNIST DDPM with CFG.

    Inputs:
        x: [B, 1, 28, 28]
        t: [B]
        y: [B], labels 0-9 or null_label
    Output:
        predicted noise with shape [B, 1, 28, 28]
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        emb_dim: int = 256,
        num_classes: int = 10,
        null_label: int = 10,
    ):
        super().__init__()
        if null_label < num_classes:
            raise ValueError("null_label must be outside the normal class range.")

        self.num_classes = num_classes
        self.null_label = null_label
        self.time_embedding = SinusoidalTimeEmbedding(emb_dim)
        self.class_embedding = nn.Embedding(null_label + 1, emb_dim)

        c = base_channels
        self.input_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        self.down1 = ResBlock(c, c, emb_dim)
        self.downsample1 = nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1)

        self.down2 = ResBlock(c, c * 2, emb_dim)
        self.downsample2 = nn.Conv2d(c * 2, c * 2, kernel_size=4, stride=2, padding=1)

        self.middle = nn.ModuleList(
            [
            ResBlock(c * 2, c * 2, emb_dim),
            ResBlock(c * 2, c * 2, emb_dim),
            ]
        )

        self.upsample1 = nn.ConvTranspose2d(c * 2, c * 2, kernel_size=4, stride=2, padding=1)
        self.up1 = ResBlock(c * 4, c, emb_dim)

        self.upsample2 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1)
        self.up2 = ResBlock(c * 2, c, emb_dim)

        self.output = nn.Sequential(
            nn.GroupNorm(num_groups=_num_groups(c), num_channels=c),
            nn.SiLU(),
            nn.Conv2d(c, in_channels, kernel_size=3, padding=1),
        )

    def _conditioning(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.ndim != 1:
            y = y.reshape(-1)
        y = y.clamp(min=0, max=self.null_label)
        return self.time_embedding(t) + self.class_embedding(y)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self._conditioning(t, y)

        h0 = self.input_conv(x)
        h1 = self.down1(h0, emb)
        h = self.downsample1(h1)

        h2 = self.down2(h, emb)
        h = self.downsample2(h2)

        for block in self.middle:
            h = block(h, emb)

        h = self.upsample1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, emb)

        h = self.upsample2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h, emb)

        return self.output(h)


# Short alias for scripts that prefer architecture-first naming.
TinyConditionalUNet = MNISTConditionalUNet
