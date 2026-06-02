from pathlib import Path
import argparse
import sys
import time

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from labs.lab4.sample_mnist_cfg import load_model as load_unet
from labs.lab_dit.sample_mnist_dit import load_dit
from src.cfg_sampler import sample_ddpm_cfg
from src.diffusion import DDPMSchedule
from src.dit.utils import count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_ckpt", type=str, default=str(ROOT / "checkpoints" / "mnist_cfg_unet.pt"))
    parser.add_argument("--dit_ckpt", type=str, default=str(ROOT / "checkpoints" / "mnist_dit.pt"))
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=16)
    parser.add_argument("--class_id", type=int, default=3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "dit" / "unet_vs_dit_samples.png"))
    parser.add_argument("--report_path", type=str, default=str(ROOT / "reports" / "unet_vs_dit_comparison.md"))
    return parser.parse_args()


@torch.no_grad()
def timed_sample(model, schedule, labels, cfg_scale, null_label, device):
    start = time.perf_counter()
    samples = sample_ddpm_cfg(
        model=model,
        shape=(labels.shape[0], 1, 28, 28),
        y=labels,
        schedule=schedule,
        cfg_scale=cfg_scale,
        null_label=null_label,
        device=device,
    )
    return samples, time.perf_counter() - start


def save_comparison(unet_samples, dit_samples, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    unet_grid = make_grid(unet_samples.detach().cpu(), nrow=4, normalize=True, value_range=(-1, 1))
    dit_grid = make_grid(dit_samples.detach().cpu(), nrow=4, normalize=True, value_range=(-1, 1))

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(unet_grid.permute(1, 2, 0).numpy(), cmap="gray")
    axes[0].set_title("U-Net DDPM")
    axes[0].axis("off")
    axes[1].imshow(dit_grid.permute(1, 2, 0).numpy(), cmap="gray")
    axes[1].set_title("Mini-DiT DDPM")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_architecture_figure(save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        "x_t image",
        "patchify",
        "patch tokens",
        "DiT blocks",
        "noise patches",
        "unpatchify",
        "eps pred",
    ]
    fig, ax = plt.subplots(figsize=(12, 2.3))
    ax.axis("off")
    for i, text in enumerate(steps):
        x = i / (len(steps) - 1)
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f4f4f4", edgecolor="#555555"),
            transform=ax.transAxes,
        )
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=((i + 0.68) / (len(steps) - 1), 0.5),
                xytext=((i + 0.32) / (len(steps) - 1), 0.5),
                arrowprops=dict(arrowstyle="->", color="#333333"),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    device = torch.device(args.device)

    unet, unet_args = load_unet(Path(args.unet_ckpt), device)
    dit, dit_args = load_dit(Path(args.dit_ckpt), device)

    timesteps = args.timesteps or min(unet_args.get("timesteps", 1000), dit_args.get("timesteps", 1000))
    schedule = DDPMSchedule(timesteps=timesteps, device=device)
    labels = torch.full((args.num_samples,), args.class_id, dtype=torch.long, device=device)

    unet_samples, unet_time = timed_sample(
        unet,
        schedule,
        labels,
        args.cfg_scale,
        unet_args.get("null_label", 10),
        device,
    )
    dit_samples, dit_time = timed_sample(
        dit,
        schedule,
        labels,
        args.cfg_scale,
        dit_args.get("null_label", 10),
        device,
    )

    save_comparison(unet_samples, dit_samples, args.save_path)
    save_architecture_figure(ROOT / "figures" / "dit" / "dit_architecture.png")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# U-Net vs DiT Comparison",
                "",
                "## Summary",
                "",
                "On MNIST, U-Net generally has a stronger convolutional inductive bias and is easier to train with small models. Mini-DiT is more architecture-general, but typically benefits from more data, more compute, and larger model scale.",
                "",
                "## Measured Smoke-Test Run",
                "",
                f"- U-Net parameters: `{count_parameters(unet):,}`",
                f"- Mini-DiT parameters: `{count_parameters(dit):,}`",
                f"- U-Net sampling time: `{unet_time:.4f}s`",
                f"- Mini-DiT sampling time: `{dit_time:.4f}s`",
                f"- CFG scale: `{args.cfg_scale}`",
                f"- Timesteps: `{timesteps}`",
                "",
                "## Interpretation",
                "",
                "U-Net processes image feature maps with convolutional locality and skip connections. DiT patchifies the image and denoises patch tokens using Transformer blocks with time/class conditioning. DiT's main advantage is scalability and a unified token modeling interface.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved comparison figure to: {args.save_path}")
    print(f"Saved comparison report to: {report_path}")


if __name__ == "__main__":
    main()
