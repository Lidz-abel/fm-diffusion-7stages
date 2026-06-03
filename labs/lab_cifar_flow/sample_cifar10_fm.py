from pathlib import Path
import argparse
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cifar10_dataset import CIFAR10_CLASSES
from src.image_flow_samplers import sample_flow
from src.image_unet import CIFAR10FlowUNet
from src.utils import save_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_fm.pt"))
    parser.add_argument("--model_type", type=str, default="unet")
    parser.add_argument("--class_id", type=int, default=3)
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--nfe", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_per_class", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "unet_fm_samples.png"))
    return parser.parse_args()


def load_unet(ckpt_path: Path, device: torch.device, use_ema: bool = False):
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = CIFAR10FlowUNet(
        in_channels=3,
        base_channels=ckpt_args.get("base_channels", 128),
        emb_dim=ckpt_args.get("time_dim", 512),
        num_classes=ckpt_args.get("num_classes", 10),
        null_label=ckpt_args.get("null_label", 10),
        use_attention=ckpt_args.get("use_attention", False),
    ).to(device)
    if use_ema and checkpoint.get("ema") is not None:
        model.load_state_dict(checkpoint["ema"]["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ckpt_args


def build_labels(args, device: torch.device) -> torch.Tensor:
    if args.all_classes:
        labels = []
        for class_id in range(len(CIFAR10_CLASSES)):
            labels.extend([class_id] * args.num_per_class)
        return torch.tensor(labels, dtype=torch.long, device=device)
    return torch.full((args.num_samples,), args.class_id, dtype=torch.long, device=device)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_unet(Path(args.ckpt), device, use_ema=args.use_ema)
    labels = build_labels(args, device)
    samples = sample_flow(
        model=model,
        shape=(labels.shape[0], 3, 32, 32),
        y=labels,
        solver=args.solver,
        nfe=args.nfe,
        cfg_scale=args.cfg_scale,
        null_label=ckpt_args.get("null_label", 10),
        device=device,
    )
    if args.all_classes:
        nrow = args.num_per_class
    else:
        nrow = max(1, int(math.sqrt(labels.shape[0])))
    save_image_grid(samples, args.save_path, nrow=nrow)
    print(f"Saved samples to: {args.save_path}")


if __name__ == "__main__":
    main()
