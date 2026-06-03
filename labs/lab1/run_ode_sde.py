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


def run_sanity_checks(figures_dir: Path, device: str) -> None:
    """
    Check two analytic facts used in this lab:
      1. Var[W_t] = t for Brownian motion.
      2. dX = -X dt has solution X_t = exp(-t) X_0.
    """
    check_steps = 200
    brownian_paths = simulate_brownian_motion(
        n_paths=20000,
        n_steps=check_steps,
        dim=1,
        device=device,
    )

    times = torch.tensor([0.25, 0.50, 0.75, 1.00], device=brownian_paths.device)
    indices = (times * check_steps).long()
    empirical_vars = brownian_paths[:, indices, 0].var(dim=0, unbiased=True)

    x0 = torch.randn(2048, 2, device=device) * 3.0
    ode_traj = euler_solver(
        x0=x0,
        vector_field=linear_drift_to_origin,
        n_steps=check_steps,
    )
    t_grid = torch.linspace(0.0, 1.0, check_steps + 1, device=x0.device)
    exact = torch.exp(-t_grid).view(-1, 1, 1) * x0.view(1, *x0.shape)
    abs_error = (ode_traj - exact).abs()

    lines = ["Stage 1 sanity checks"]
    lines.append("Brownian motion variance check: Var[W_t] should be close to t.")
    for t_value, empirical_var in zip(times.tolist(), empirical_vars.tolist()):
        lines.append(f"  t={t_value:.2f}: empirical_var={empirical_var:.6f}, target={t_value:.6f}")

    lines.append("ODE check for dX=-X dt: Euler solution should approximate exp(-t) X0.")
    lines.append(f"  max_abs_error={abs_error.max().item():.6f}")
    lines.append(f"  mean_abs_error={abs_error.mean().item():.6f}")

    report = "\n".join(lines)
    (figures_dir / "stage1_sanity_checks.txt").write_text(report + "\n", encoding="utf-8")
    print(report)


def main() -> None:
    device = "cpu"
    batch = 100
    n_steps = 100

    figures_dir = ROOT / "figures" / "stage1"
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_sanity_checks(figures_dir, device=device)

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
