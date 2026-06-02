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


def save_labeled_image_rows(
    image_rows: list[torch.Tensor],
    row_labels: list[str],
    col_labels: list[str],
    save_path: str | Path,
    title: str,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """
    Save a compact labeled image grid.

    Each tensor in image_rows must have shape [num_cols, C, H, W].
    """
    import matplotlib.pyplot as plt

    if len(image_rows) != len(row_labels):
        raise ValueError("image_rows and row_labels must have the same length.")
    if not image_rows:
        raise ValueError("image_rows must be non-empty.")

    num_rows = len(image_rows)
    num_cols = len(col_labels)
    for row in image_rows:
        if row.shape[0] != num_cols:
            raise ValueError(f"Expected {num_cols} images per row, got {row.shape[0]}.")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(1.35 * num_cols, 1.55 * num_rows + 0.8),
        squeeze=False,
    )
    low, high = value_range
    for row_idx, row in enumerate(image_rows):
        images = row.detach().cpu().float()
        images = (images - low) / (high - low)
        images = images.clamp(0.0, 1.0)
        for col_idx in range(num_cols):
            ax = axes[row_idx][col_idx]
            image = images[col_idx].permute(1, 2, 0).numpy()
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
