import math
import torch


def sample_standard_normal(
    n: int,
    dim: int = 2,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample from standard Gaussian N(0, I).

    Args:
        n: number of samples
        dim: data dimension
        device: torch device

    Returns:
        Tensor with shape [n, dim]
    """
    return torch.randn(n, dim, device=device)


def sample_gaussian_mixture(
    n: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Sample 2D points from an 8-component Gaussian mixture.

    Args:
        n: number of samples
        device: torch device

    Returns:
        Tensor with shape [n, 2]
    """
    angles = torch.linspace(0, 2 * math.pi, 9, device=device)[:-1]
    centers = torch.stack(
        [torch.cos(angles), torch.sin(angles)],
        dim=1,
    ) * 3.0

    idx = torch.randint(0, len(centers), (n,), device=device)
    x = centers[idx] + 0.25 * torch.randn(n, 2, device=device)
    return x


def sample_two_moons_like(
    n: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    A simple two-moons-like toy distribution implemented without sklearn.
    """
    n1 = n // 2
    n2 = n - n1

    theta1 = torch.rand(n1, device=device) * math.pi
    moon1 = torch.stack([torch.cos(theta1), torch.sin(theta1)], dim=1)

    theta2 = torch.rand(n2, device=device) * math.pi
    moon2 = torch.stack([1.0 - torch.cos(theta2), -torch.sin(theta2) - 0.5], dim=1)

    x = torch.cat([moon1, moon2], dim=0)
    x = x * 2.0 + 0.08 * torch.randn_like(x)
    return x