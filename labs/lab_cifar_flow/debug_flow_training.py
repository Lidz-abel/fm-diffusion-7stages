from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.image_flow_matching import flow_matching_loss
from src.image_unet import CIFAR10FlowUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="shape_test", choices=["shape_test", "loss_backward"])
    parser.add_argument("--model", type=str, default="unet_v2")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = CIFAR10FlowUNet(
        in_channels=3,
        base_channels=args.base_channels,
        emb_dim=args.time_dim,
        num_classes=10,
        null_label=10,
        use_attention=args.model == "unet_v2",
    ).to(device)
    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    t = torch.rand(args.batch_size, device=device)
    y = torch.arange(args.batch_size, device=device) % 10
    out = model(x, t, y)
    print(f"input:  {tuple(x.shape)}")
    print(f"output: {tuple(out.shape)}")
    if out.shape != x.shape:
        raise RuntimeError(f"Output shape {out.shape} does not match input shape {x.shape}.")

    if args.mode == "loss_backward":
        loss = flow_matching_loss(model, x, y, null_label=10, drop_label_prob=0.1)
        loss.backward()
        print(f"loss_backward ok: {float(loss.item()):.6f}")


if __name__ == "__main__":
    main()
