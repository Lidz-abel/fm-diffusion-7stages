from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.ode import euler_solver
from src.sde import euler_maruyama_solver, simulate_brownian_motion
from src.visualization import plot_trajectories, plot_brownian_paths


def linear_drift_to_origin(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Drift field:
        u_t(x) = -x

    This pulls samples toward the origin.
    """
    return -x


def constant_sigma(t: torch.Tensor) -> float:
    """
    Constant diffusion coefficient.
    """
    return 0.5


def main() -> None:
    device = "cpu"
    batch = 100
    n_steps = 100

    figures_dir = ROOT / "figures" / "stage1"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Brownian motion
    brownian_paths = simulate_brownian_motion(
        n_paths=20,
        n_steps=n_steps,
        dim=1,
        device=device,
    )

    plot_brownian_paths(
        brownian_paths,
        title="Brownian motion paths",
        save_path=str(figures_dir / "brownian_motion.png"),
    )

    # Initial points
    x0 = torch.randn(batch, 2, device=device) * 3.0

    # ODE trajectories
    ode_traj = euler_solver(
        x0=x0,
        vector_field=linear_drift_to_origin,
        n_steps=n_steps,
    )

    # SDE trajectories
    sde_traj = euler_maruyama_solver(
        x0=x0,
        drift=linear_drift_to_origin,
        sigma=constant_sigma,
        n_steps=n_steps,
    )

    plot_trajectories(
        ode_traj,
        title="ODE trajectories: dX = -X dt",
        save_path=str(figures_dir / "ode_trajectories.png"),
    )

    plot_trajectories(
        sde_traj,
        title="SDE trajectories: dX = -X dt + sigma dW",
        save_path=str(figures_dir / "sde_trajectories.png"),
    )

    print(f"Saved ODE/SDE figures to: {figures_dir}")


if __name__ == "__main__":
    main()