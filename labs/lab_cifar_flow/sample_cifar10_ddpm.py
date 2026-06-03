from pathlib import Path
import argparse
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cfg_sampler import sample_ddpm_cfg
from src.cifar10_dataset import CIFAR10_CLASSES
from src.diffusion import DDPMSchedule
from src.image_unet import CIFAR10FlowUNet
from src.utils import save_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_ddpm_strong.pt"))
    parser.add_argument("--class_id", type=int, default=3)
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_per_class", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "final_ddpm_strong_cfg2.png"))
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device, use_ema: bool):
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = CIFAR10FlowUNet(
        in_channels=3,
        base_channels=ckpt_args.get("base_channels", 192),
        emb_dim=ckpt_args.get("time_dim", 512),
        num_classes=ckpt_args.get("num_classes", 10),
        null_label=ckpt_args.get("null_label", 10),
        use_attention=ckpt_args.get("use_attention", True),
        time_scale=ckpt_args.get("time_scale", 1.0),
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
    model, ckpt_args = load_model(Path(args.ckpt), device, use_ema=args.use_ema)
    timesteps = args.timesteps or ckpt_args.get("timesteps", 1000)
    schedule = DDPMSchedule(
        timesteps=timesteps,
        beta_start=ckpt_args.get("beta_start", 1e-4),
        beta_end=ckpt_args.get("beta_end", 2e-2),
        device=device,
    )
    labels = build_labels(args, device)
    samples = sample_ddpm_cfg(
        model=model,
        shape=(labels.shape[0], 3, 32, 32),
        y=labels,
        schedule=schedule,
        cfg_scale=args.cfg_scale,
        null_label=ckpt_args.get("null_label", 10),
        device=device,
    )
    nrow = args.num_per_class if args.all_classes else max(1, int(math.sqrt(labels.shape[0])))
    save_image_grid(samples.clamp(-1.0, 1.0), args.save_path, nrow=nrow)
    print(f"Saved DDPM samples to: {args.save_path}")


if __name__ == "__main__":
    main()
