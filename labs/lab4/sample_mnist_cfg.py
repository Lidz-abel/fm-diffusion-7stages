from pathlib import Path
import argparse
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cfg_sampler import sample_ddpm_cfg
from src.diffusion import DDPMSchedule
from src.unet import MNISTConditionalUNet
from src.utils import save_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "mnist_cfg_unet.pt"))
    parser.add_argument("--class_id", type=int, default=3)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=None)
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = MNISTConditionalUNet(
        in_channels=1,
        base_channels=ckpt_args.get("base_channels", 64),
        emb_dim=ckpt_args.get("time_dim", 256),
        num_classes=ckpt_args.get("num_classes", 10),
        null_label=ckpt_args.get("null_label", 10),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ckpt_args


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    model, ckpt_args = load_model(ckpt_path, device)

    timesteps = args.timesteps or ckpt_args.get("timesteps", 1000)
    null_label = ckpt_args.get("null_label", 10)
    schedule = DDPMSchedule(timesteps=timesteps, device=device)

    labels = torch.full((args.num_samples,), args.class_id, device=device, dtype=torch.long)
    samples = sample_ddpm_cfg(
        model=model,
        shape=(args.num_samples, 1, 28, 28),
        y=labels,
        schedule=schedule,
        cfg_scale=args.cfg_scale,
        null_label=null_label,
        device=device,
    )

    save_path = args.save_path
    if save_path is None:
        scale_text = str(args.cfg_scale).replace(".", "p")
        save_path = ROOT / "figures" / "day4" / f"samples_class{args.class_id}_scale{scale_text}.png"

    save_image_grid(samples, save_path, nrow=int(math.sqrt(args.num_samples)))
    print(f"Saved samples to: {save_path}")


if __name__ == "__main__":
    main()
