from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt


def _ensure_parent(save_path: Optional[str]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def plot_points(
    x: torch.Tensor,
    title: str,
    save_path: Optional[str] = None,
    s: float = 3.0,
) -> None:
    """
    Plot 2D point cloud.
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"Expected x with shape [n, 2], got {x.shape}.")

    x_np = x.detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.scatter(x_np[:, 0], x_np[:, 1], s=s, alpha=0.6)
    plt.axis("equal")
    plt.title(title)
    plt.grid(alpha=0.2)

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_trajectories(
    traj: torch.Tensor,
    title: str,
    save_path: Optional[str] = None,
    max_paths: int = 50,
) -> None:
    """
    Plot 2D trajectories.

    Args:
        traj: Tensor with shape [n_steps + 1, batch, 2]
    """
    if traj.ndim != 3 or traj.shape[-1] != 2:
        raise ValueError(f"Expected traj with shape [steps, batch, 2], got {traj.shape}.")

    traj_np = traj[:, :max_paths].detach().cpu().numpy()

    plt.figure(figsize=(5, 5))
    for i in range(traj_np.shape[1]):
        plt.plot(traj_np[:, i, 0], traj_np[:, i, 1], linewidth=0.8, alpha=0.8)

    plt.axis("equal")
    plt.title(title)
    plt.grid(alpha=0.2)

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_brownian_paths(
    paths: torch.Tensor,
    title: str,
    save_path: Optional[str] = None,
    max_paths: int = 10,
) -> None:
    """
    Plot 1D Brownian paths.

    Args:
        paths: Tensor with shape [n_paths, n_steps + 1, 1]
    """
    if paths.ndim != 3 or paths.shape[-1] != 1:
        raise ValueError(f"Expected paths with shape [n_paths, steps, 1], got {paths.shape}.")

    y = paths[:max_paths, :, 0].detach().cpu().numpy()

    plt.figure(figsize=(7, 4))
    for i in range(y.shape[0]):
        plt.plot(y[i], linewidth=1.0)

    plt.title(title)
    plt.xlabel("step")
    plt.ylabel("W_t")
    plt.grid(alpha=0.2)

    _ensure_parent(save_path)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_nfe_samples(
    samples_by_nfe: dict[int, torch.Tensor],
    save_path: str,
) -> None:
    """
    Plot samples generated with different NFE values.
    """
    n_cols = len(samples_by_nfe)
    plt.figure(figsize=(4 * n_cols, 4))

    for idx, (nfe, samples) in enumerate(samples_by_nfe.items(), start=1):
        x_np = samples.detach().cpu().numpy()

        plt.subplot(1, n_cols, idx)
        plt.scatter(x_np[:, 0], x_np[:, 1], s=3, alpha=0.6)
        plt.axis("equal")
        plt.title(f"NFE={nfe}")
        plt.grid(alpha=0.2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def plot_vector_field(
    model,
    t_value: float,
    save_path: str,
    xlim: tuple[float, float] = (-4.5, 4.5),
    ylim: tuple[float, float] = (-4.5, 4.5),
    grid_size: int = 25,
    device: str = "cpu",
) -> None:
    """
    Visualize learned vector field v_theta(x, t) at a fixed time t.
    """
    xs = torch.linspace(xlim[0], xlim[1], grid_size, device=device)
    ys = torch.linspace(ylim[0], ylim[1], grid_size, device=device)

    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    t = torch.full((points.shape[0], 1), t_value, device=device)
    velocity = model(points, t)

    p = points.detach().cpu().numpy()
    v = velocity.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.quiver(
        p[:, 0],
        p[:, 1],
        v[:, 0],
        v[:, 1],
        angles="xy",
        scale_units="xy",
        scale=20,
        width=0.003,
    )
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.axis("equal")
    plt.title(f"Learned vector field at t={t_value:.2f}")
    plt.grid(alpha=0.2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_sample_panels(
    samples_by_title: dict[str, torch.Tensor],
    save_path: str,
    s: float = 3.0,
) -> None:
    """
    Plot several 2D sample clouds side by side.
    """
    n_cols = len(samples_by_title)
    plt.figure(figsize=(4 * n_cols, 4))

    for idx, (title, samples) in enumerate(samples_by_title.items(), start=1):
        if samples.ndim != 2 or samples.shape[1] != 2:
            raise ValueError(f"Expected samples with shape [n, 2], got {samples.shape}.")
        x_np = samples.detach().cpu().numpy()

        plt.subplot(1, n_cols, idx)
        plt.scatter(x_np[:, 0], x_np[:, 1], s=s, alpha=0.6)
        plt.axis("equal")
        plt.title(title)
        plt.grid(alpha=0.2)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
