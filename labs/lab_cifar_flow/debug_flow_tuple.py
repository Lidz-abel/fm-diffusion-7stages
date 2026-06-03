from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cifar10_dataset import get_cifar10_dataloader
from src.image_flow_matching import sample_rectified_flow_tuple
from src.utils import save_image_grid, save_labeled_image_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--save_dir", type=str, default=str(ROOT / "figures" / "cifar_flow" / "debug"))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataloader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        train=True,
        root=args.data_dir,
        num_workers=args.num_workers,
        download=True,
    )
    x1, _ = next(iter(dataloader))
    x1 = x1.to(device)
    batch = sample_rectified_flow_tuple(x1)

    save_image_grid(x1[:16], save_dir / "debug_real_images.png", nrow=4)
    save_image_grid(batch.x0[:16], save_dir / "debug_noise_images.png", nrow=4)

    t_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for t_value in t_values:
        xt = (1.0 - t_value) * batch.x0[:10] + t_value * batch.x1[:10]
        rows.append(xt)
    save_labeled_image_rows(
        rows,
        row_labels=[f"t={t:g}" for t in t_values],
        col_labels=[str(i) for i in range(10)],
        save_path=save_dir / "debug_xt_interpolation.png",
        title="Rectified Flow x_t = (1-t)x0 + t*x1",
    )

    target_v = batch.target_v.detach().float().cpu()
    stats = {
        "x1_min": float(x1.min().item()),
        "x1_max": float(x1.max().item()),
        "x0_mean": float(batch.x0.mean().item()),
        "x0_std": float(batch.x0.std(unbiased=False).item()),
        "target_v_mean": float(target_v.mean().item()),
        "target_v_std": float(target_v.std(unbiased=False).item()),
        "target_v_min": float(target_v.min().item()),
        "target_v_max": float(target_v.max().item()),
    }
    with (save_dir / "debug_velocity_stats.txt").open("w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value:.6f}\n")

    print(f"Saved debug figures and stats to: {save_dir}")


if __name__ == "__main__":
    main()
