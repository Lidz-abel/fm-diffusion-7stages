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
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_fm_v2.pt"))
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--nfe", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "ema_vs_raw.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    labels = torch.arange(10, dtype=torch.long, device=device)

    raw_model, ckpt_args = load_unet(Path(args.ckpt), device, use_ema=False)
    raw = sample_flow(
        raw_model,
        shape=(10, 3, 32, 32),
        y=labels,
        solver=args.solver,
        nfe=args.nfe,
        cfg_scale=args.cfg_scale,
        null_label=ckpt_args.get("null_label", 10),
        device=device,
    )

    ema_model, _ = load_unet(Path(args.ckpt), device, use_ema=True)
    torch.manual_seed(args.seed)
    ema = sample_flow(
        ema_model,
        shape=(10, 3, 32, 32),
        y=labels,
        solver=args.solver,
        nfe=args.nfe,
        cfg_scale=args.cfg_scale,
        null_label=ckpt_args.get("null_label", 10),
        device=device,
    )

    save_labeled_image_rows(
        [raw, ema],
        row_labels=["raw", "EMA"],
        col_labels=CIFAR10_CLASSES,
        save_path=args.save_path,
        title=f"EMA vs Raw | solver={args.solver}, NFE={args.nfe}, cfg={args.cfg_scale}",
    )
    print(f"Saved EMA comparison to: {args.save_path}")


if __name__ == "__main__":
    main()
