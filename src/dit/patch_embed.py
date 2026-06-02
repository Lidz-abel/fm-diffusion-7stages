import torch
import torch.nn as nn


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert images to flattened non-overlapping patches.

    Args:
        x: [B, C, H, W]

    Returns:
        patches: [B, N, C * patch_size * patch_size]
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x with shape [B, C, H, W], got {x.shape}.")

    batch, channels, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("height and width must be divisible by patch_size.")

    patches = x.reshape(
        batch,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(batch, -1, channels * patch_size * patch_size)


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    image_size: int,
    channels: int,
) -> torch.Tensor:
    """
    Reconstruct images from flattened non-overlapping patches.

    Args:
        patches: [B, N, C * patch_size * patch_size]

    Returns:
        x: [B, C, image_size, image_size]
    """
    if patches.ndim != 3:
        raise ValueError(f"Expected patches with shape [B, N, patch_dim], got {patches.shape}.")

    batch, num_patches, patch_dim = patches.shape
    patches_per_side = image_size // patch_size
    expected_patches = patches_per_side * patches_per_side
    expected_dim = channels * patch_size * patch_size

    if num_patches != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {num_patches}.")
    if patch_dim != expected_dim:
        raise ValueError(f"Expected patch_dim={expected_dim}, got {patch_dim}.")

    x = patches.reshape(
        batch,
        patches_per_side,
        patches_per_side,
        channels,
        patch_size,
        patch_size,
    )
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(batch, channels, image_size, image_size)


class PatchEmbed(nn.Module):
    """
    Patchify images and linearly project patches to token embeddings.
    """

    def __init__(
        self,
        image_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        hidden_dim: int = 192,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.patch_dim = in_channels * patch_size * patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Linear(self.patch_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = patchify(x, self.patch_size)
        return self.proj(patches)
