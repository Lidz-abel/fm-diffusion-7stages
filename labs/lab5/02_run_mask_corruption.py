from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.discrete_diffusion import MaskCorruption, MaskDiffusionSchedule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "day5" / "mask_corruption_process.png"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def format_sequence(seq: torch.Tensor) -> str:
    return "[" + ", ".join("MASK" if int(x) == 0 else str(int(x)) for x in seq) + "]"


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    schedule = MaskDiffusionSchedule(num_steps=args.num_steps)
    corruption = MaskCorruption(schedule=schedule, mask_token_id=0)

    x0 = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    t_values = [0, args.num_steps // 4, args.num_steps // 2, (3 * args.num_steps) // 4, args.num_steps - 1]

    rows = []
    for t_value in t_values:
        t = torch.full((1,), t_value, dtype=torch.long)
        xt, _ = corruption.q_sample(x0, t)
        rows.append((t_value, xt[0].clone()))
        print(f"t={t_value}: {format_sequence(xt[0])}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    table_data = [[f"t={t}", format_sequence(seq)] for t, seq in rows]
    table = ax.table(
        cellText=table_data,
        colLabels=["time", "corrupted sequence"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax.set_title("Mask corruption process")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mask corruption figure to: {save_path}")


if __name__ == "__main__":
    main()
