import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embedding import SinusoidalTimeEmbedding


def _num_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ImageResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        attended = attended.transpose(1, 2).reshape(b, c, h, w)
        return x + attended


class CIFAR10FlowUNet(nn.Module):
    """
    Class-conditional U-Net velocity network for 32x32 RGB Rectified Flow.

    Inputs:
        x: [B, 3, 32, 32]
        t: [B] in [0, 1]
        y: [B], labels 0-9 or null_label
    Output:
        velocity prediction with shape [B, 3, 32, 32]
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        emb_dim: int = 512,
        num_classes: int = 10,
        null_label: int = 10,
        use_attention: bool = False,
        time_scale: float = 1000.0,
    ):
        super().__init__()
        if null_label < num_classes:
            raise ValueError("null_label must be outside the normal class range.")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.null_label = null_label
        self.use_attention = use_attention
        self.time_scale = time_scale

        self.time_embedding = SinusoidalTimeEmbedding(emb_dim)
        self.class_embedding = nn.Embedding(null_label + 1, emb_dim)

        c = base_channels
        self.input_conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)

        self.down1 = ImageResBlock(c, c, emb_dim)
        self.downsample1 = nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1)

        self.down2 = ImageResBlock(c, c * 2, emb_dim)
        self.attn16 = SelfAttention2d(c * 2) if use_attention else nn.Identity()
        self.downsample2 = nn.Conv2d(c * 2, c * 2, kernel_size=4, stride=2, padding=1)

        self.down3 = ImageResBlock(c * 2, c * 4, emb_dim)
        self.attn8 = SelfAttention2d(c * 4) if use_attention else nn.Identity()
        self.downsample3 = nn.Conv2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1)

        self.middle1 = ImageResBlock(c * 4, c * 4, emb_dim)
        self.middle_attn = SelfAttention2d(c * 4) if use_attention else nn.Identity()
        self.middle2 = ImageResBlock(c * 4, c * 4, emb_dim)

        self.upsample1 = nn.ConvTranspose2d(c * 4, c * 4, kernel_size=4, stride=2, padding=1)
        self.up1 = ImageResBlock(c * 8, c * 2, emb_dim)

        self.upsample2 = nn.ConvTranspose2d(c * 2, c * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = ImageResBlock(c * 4, c, emb_dim)

        self.upsample3 = nn.ConvTranspose2d(c, c, kernel_size=4, stride=2, padding=1)
        self.up3 = ImageResBlock(c * 2, c, emb_dim)

        self.output = nn.Sequential(
            nn.GroupNorm(_num_groups(c), c),
            nn.SiLU(),
            nn.Conv2d(c, in_channels, kernel_size=3, padding=1),
        )

    def _conditioning(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if y.ndim != 1:
            y = y.reshape(-1)
        y = y.clamp(min=0, max=self.null_label)
        return self.time_embedding(t * self.time_scale) + self.class_embedding(y)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self._conditioning(t, y)

        h0 = self.input_conv(x)
        h1 = self.down1(h0, emb)
        h = self.downsample1(h1)

        h2 = self.down2(h, emb)
        h2 = self.attn16(h2)
        h = self.downsample2(h2)

        h3 = self.down3(h, emb)
        h3 = self.attn8(h3)
        h = self.downsample3(h3)

        h = self.middle1(h, emb)
        h = self.middle_attn(h)
        h = self.middle2(h, emb)

        h = self.upsample1(h)
        h = self.up1(torch.cat([h, h3], dim=1), emb)

        h = self.upsample2(h)
        h = self.up2(torch.cat([h, h2], dim=1), emb)

        h = self.upsample3(h)
        h = self.up3(torch.cat([h, h1], dim=1), emb)
        return self.output(h)
