from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class VisualDatasetInfo:
    num_classes: int
    class_names: list[str]


def _image_transform(image_size: int, augment: bool) -> transforms.Compose:
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


def get_visual_dataloader(
    dataset: str,
    root: str | Path,
    image_size: int = 64,
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 4,
    download: bool = True,
    augment: bool | None = None,
    shuffle: bool | None = None,
) -> tuple[DataLoader, VisualDatasetInfo]:
    """
    Generic visual dataloader for the 64x64 showcase branch.

    Supported datasets:
    - imagefolder: root must contain class subdirectories.
    - pets: Oxford-IIIT Pet, target is category id.
    """
    if augment is None:
        augment = train
    if shuffle is None:
        shuffle = train

    dataset = dataset.lower()
    transform = _image_transform(image_size=image_size, augment=augment)
    root = Path(root)

    if dataset == "imagefolder":
        ds = datasets.ImageFolder(root=str(root), transform=transform)
        info = VisualDatasetInfo(num_classes=len(ds.classes), class_names=list(ds.classes))
    elif dataset == "pets":
        split = "trainval" if train else "test"
        ds = datasets.OxfordIIITPet(
            root=str(root),
            split=split,
            target_types="category",
            transform=transform,
            download=download,
        )
        class_names = list(getattr(ds, "classes", [f"pet_{idx}" for idx in range(37)]))
        info = VisualDatasetInfo(num_classes=len(class_names), class_names=class_names)
    else:
        raise ValueError("dataset must be 'imagefolder' or 'pets'.")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
    return loader, info
