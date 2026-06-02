from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.theory_plotting import save_continuous_vs_discrete, save_unified_framework


def main():
    figures_dir = ROOT / "figures" / "day5"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_unified_framework(figures_dir / "final_unified_framework.png")
    save_continuous_vs_discrete(figures_dir / "continuous_vs_discrete_diffusion.png")
    print(f"Saved Stage 5 theory figures to: {figures_dir}")


if __name__ == "__main__":
    main()
