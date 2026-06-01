from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_dataloader(
    batch_size: int,
    train: bool = True,
    root: str | Path = "data",
    num_workers: int = 2,
    download: bool = True,
) -> DataLoader:
    """
    MNIST dataloader with images scaled to [-1, 1].
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=str(root),
        train=train,
        transform=transform,
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
