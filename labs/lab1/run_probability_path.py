from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.toy_data import sample_gaussian_mixture, sample_standard_normal
from src.visualization import plot_points


def main() -> None:
    device = "cpu"
    n = 5000

    figures_dir = ROOT / "figures" / "stage1"
    figures_dir.mkdir(parents=True, exist_ok=True)

    z = sample_gaussian_mixture(n, device=device)
    eps = sample_standard_normal(n, dim=2, device=device)

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x_t = t * z + (1.0 - t) * eps

        plot_points(
            x_t,
            title=f"Probability path: t={t:.2f}",
            save_path=str(figures_dir / f"probability_path_t_{t:.2f}.png"),
        )

    print(f"Saved probability path figures to: {figures_dir}")


if __name__ == "__main__":
    main()