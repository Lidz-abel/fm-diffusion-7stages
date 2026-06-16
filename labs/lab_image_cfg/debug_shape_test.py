from __future__ import annotations

from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.class_cond_unet import ClassConditionalUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--null_label", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--attention_resolutions", type=int, nargs="*", default=[32, 16])
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    null_label = args.num_classes if args.null_label is None else args.null_label
    model = ClassConditionalUNet(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        num_classes=args.num_classes,
        null_label=null_label,
        use_attention=bool(args.attention_resolutions),
        time_scale=1.0,
        image_size=args.image_size,
        attention_resolutions=args.attention_resolutions,
    ).to(device)
    x = torch.randn(
        args.batch_size,
        args.in_channels,
        args.image_size,
        args.image_size,
        device=device,
    )
    t = torch.randint(0, 1000, (args.batch_size,), device=device)
    y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
    y_null = torch.full_like(y, null_label)

    out_cond = model(x, t, y)
    out_uncond = model(x, t, y_null)
    if out_cond.shape != x.shape:
        raise RuntimeError(f"conditional output shape {out_cond.shape} != input shape {x.shape}")
    if out_uncond.shape != x.shape:
        raise RuntimeError(f"unconditional output shape {out_uncond.shape} != input shape {x.shape}")

    print(f"device: {device}")
    print(f"input: {tuple(x.shape)}")
    print(f"conditional output: {tuple(out_cond.shape)}")
    print(f"unconditional output: {tuple(out_uncond.shape)}")
    print(f"num_classes: {args.num_classes}")
    print(f"null_label: {null_label}")
    print("shape test passed")


if __name__ == "__main__":
    main()
