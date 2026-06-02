from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure_dir", type=str, default=str(ROOT / "figures" / "cifar_flow"))
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "phase2_summary.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    figure_dir = Path(args.figure_dir)
    panels = [
        ("NFE Ablation", figure_dir / "nfe_ablation_unet.png"),
        ("Solver Ablation", figure_dir / "solver_ablation_unet.png"),
        ("Guidance Ablation", figure_dir / "guidance_ablation_unet.png"),
        ("Runtime vs NFE", figure_dir / "runtime_vs_nfe.png"),
    ]
    existing = [(title, path) for title, path in panels if path.exists()]
    if not existing:
        raise FileNotFoundError(f"No source figures found in {figure_dir}.")

    fig, axes = plt.subplots(len(existing), 1, figsize=(12, 4 * len(existing)))
    if len(existing) == 1:
        axes = [axes]

    for ax, (title, path) in zip(axes, existing):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
    print(f"Saved Phase 2 summary figure to: {save_path}")


if __name__ == "__main__":
    main()
