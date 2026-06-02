from pathlib import Path
import argparse
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cfg_sampler import sample_ddpm_cfg
from src.diffusion import DDPMSchedule
from src.dit.model import MiniDiT
from src.dit.utils import dit_checkpoint_args
from src.utils import save_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "mnist_dit.pt"))
    parser.add_argument("--class_id", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "dit" / "mnist_dit_samples.png"))
    return parser.parse_args()


def load_dit(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = MiniDiT(**dit_checkpoint_args(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("args", {})


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, ckpt_args = load_dit(Path(args.ckpt), device)
    timesteps = args.timesteps or ckpt_args.get("timesteps", 1000)
    schedule = DDPMSchedule(timesteps=timesteps, device=device)

    labels = torch.full((args.num_samples,), args.class_id, dtype=torch.long, device=device)
    samples = sample_ddpm_cfg(
        model=model,
        shape=(args.num_samples, 1, 28, 28),
        y=labels,
        schedule=schedule,
        cfg_scale=args.cfg_scale,
        null_label=ckpt_args.get("null_label", 10),
        device=device,
    )
    save_image_grid(samples, args.save_path, nrow=max(1, int(math.sqrt(args.num_samples))))
    print(f"Saved MiniDiT samples to: {args.save_path}")


if __name__ == "__main__":
    main()
