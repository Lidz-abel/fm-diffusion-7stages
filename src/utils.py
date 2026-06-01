from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image


def save_image_grid(
    images: torch.Tensor,
    save_path: str | Path,
    nrow: int = 8,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """
    Save a grid of image tensors.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(
        images.detach().cpu(),
        nrow=nrow,
        normalize=True,
        value_range=value_range,
    )
    save_image(grid, save_path)
