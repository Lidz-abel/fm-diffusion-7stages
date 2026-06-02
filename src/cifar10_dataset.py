from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_cifar10_dataloader(
    batch_size: int,
    train: bool = True,
    root: str | Path = "data",
    num_workers: int = 4,
    download: bool = True,
) -> DataLoader:
    """
    CIFAR-10 dataloader with images scaled to [-1, 1].
    """
    transform_list = []
    if train:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = datasets.CIFAR10(
        root=str(root),
        train=train,
        transform=transforms.Compose(transform_list),
        download=download,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
    )
