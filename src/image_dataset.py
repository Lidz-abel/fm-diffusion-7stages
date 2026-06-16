from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .cifar10_dataset import CIFAR10_CLASSES


@dataclass
class ImageDatasetInfo:
    num_classes: int
    class_names: list[str]


class SyntheticClassImageDataset(Dataset):
    """
    Small random image dataset for shape and training-loop smoke tests.
    """

    def __init__(self, num_samples: int, image_size: int, in_channels: int, num_classes: int):
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.randn(self.in_channels, self.image_size, self.image_size).clamp(-1.0, 1.0)
        label = torch.tensor(index % self.num_classes, dtype=torch.long)
        return image, label


def image_transform(image_size: int, train: bool, augment: bool | None = None) -> transforms.Compose:
    if augment is None:
        augment = train
    transform_list = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if augment:
        transform_list.append(transforms.RandomHorizontalFlip())
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(transform_list)


def get_image_dataloader(
    dataset: str,
    data_dir: str | Path,
    image_size: int,
    batch_size: int,
    train: bool = True,
    num_workers: int = 4,
    download: bool = True,
    augment: bool | None = None,
    shuffle: bool | None = None,
    in_channels: int = 3,
    num_classes: int | None = None,
    synthetic_samples: int = 128,
) -> tuple[DataLoader, ImageDatasetInfo]:
    """
    Generic image dataloader for class-conditional experiments.

    Supported datasets:
    - cifar10
    - imagefolder
    - synthetic, for smoke tests only
    """
    dataset = dataset.lower()
    data_dir = Path(data_dir)
    if shuffle is None:
        shuffle = train

    if dataset == "cifar10":
        transform_list = []
        use_augment = augment if augment is not None else train
        if use_augment:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        ds = datasets.CIFAR10(
            root=str(data_dir),
            train=train,
            transform=transforms.Compose(transform_list),
            download=download,
        )
        info = ImageDatasetInfo(num_classes=10, class_names=list(CIFAR10_CLASSES))
    elif dataset == "imagefolder":
        ds = datasets.ImageFolder(
            root=str(data_dir),
            transform=image_transform(image_size=image_size, train=train, augment=augment),
        )
        info = ImageDatasetInfo(num_classes=len(ds.classes), class_names=list(ds.classes))
    elif dataset == "synthetic":
        if num_classes is None:
            num_classes = 10
        ds = SyntheticClassImageDataset(
            num_samples=synthetic_samples,
            image_size=image_size,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        info = ImageDatasetInfo(
            num_classes=num_classes,
            class_names=[f"class_{idx:03d}" for idx in range(num_classes)],
        )
    else:
        raise ValueError("dataset must be one of: cifar10, imagefolder, synthetic.")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
    return loader, info
