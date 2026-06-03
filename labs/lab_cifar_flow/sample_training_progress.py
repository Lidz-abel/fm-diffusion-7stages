from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from labs.lab_cifar_flow.sample_cifar10_fm import load_unet
from src.cifar10_dataset import CIFAR10_CLASSES
from src.image_flow_samplers import sample_flow
from src.utils import save_labeled_image_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=str, nargs="+", required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "training_progress_grid.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.ckpts):
        raise ValueError("--labels must match --ckpts length.")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    labels = torch.arange(10, dtype=torch.long, device=device)
    rows = []
    row_labels = []

    for idx, ckpt in enumerate(args.ckpts):
        torch.manual_seed(args.seed)
        model, ckpt_args = load_unet(Path(ckpt), device, use_ema=args.use_ema)
        samples = sample_flow(
            model,
            shape=(10, 3, 32, 32),
            y=labels,
            solver=args.solver,
            nfe=args.nfe,
            cfg_scale=args.cfg_scale,
            null_label=ckpt_args.get("null_label", 10),
            device=device,
        )
        rows.append(samples)
        row_labels.append(args.labels[idx] if args.labels else Path(ckpt).stem)

    save_labeled_image_rows(
        rows,
        row_labels=row_labels,
        col_labels=CIFAR10_CLASSES,
        save_path=args.save_path,
        title=f"Training Progress | EMA={args.use_ema}, solver={args.solver}, NFE={args.nfe}",
    )
    print(f"Saved training progress grid to: {args.save_path}")


if __name__ == "__main__":
    main()
