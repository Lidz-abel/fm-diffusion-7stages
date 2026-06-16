from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from .image_dataset import ImageDatasetInfo, get_image_dataloader


def get_imagenet128_subset_dataloader(
    root: str | Path,
    image_size: int = 128,
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 8,
    augment: bool | None = None,
    shuffle: bool | None = None,
) -> tuple[DataLoader, ImageDatasetInfo]:
    """
    ImageNet-128 subset loader.

    The expected layout is ImageFolder-compatible:

        root/train/class_xxx/*.jpg
        root/val/class_xxx/*.jpg

    If root already points to the split directory, it is used directly.
    """
    root = Path(root)
    split_dir = root / ("train" if train else "val")
    data_dir = split_dir if split_dir.exists() else root
    return get_image_dataloader(
        dataset="imagefolder",
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        train=train,
        num_workers=num_workers,
        download=False,
        augment=augment,
        shuffle=shuffle,
    )
