from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from labs.lab4.sample_mnist_cfg import load_model
from src.cfg_sampler import sample_ddpm_cfg
from src.diffusion import DDPMSchedule
from src.utils import save_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "mnist_cfg_unet.pt"))
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 1, 2, 4, 7])
    parser.add_argument("--num_per_class", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "day4" / "cfg_scale_comparison.png"))
    return parser.parse_args()


def save_cfg_comparison(
    samples_by_scale: list[tuple[float, torch.Tensor]],
    save_path: str | Path,
    num_per_class: int,
) -> None:
    rows = []
    for _, samples in samples_by_scale:
        rows.append(samples.detach().cpu())
    all_samples = torch.cat(rows, dim=0)
    grid = make_grid(
        all_samples,
        nrow=10 * num_per_class,
        normalize=True,
        value_range=(-1, 1),
        padding=2,
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(2 * len(samples_by_scale), 10))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.title("Rows: CFG scales, columns: digits 0-9")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_latent_diffusion_pipeline(save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    boxes = [
        "image x",
        "encoder",
        "latent z",
        "diffusion U-Net",
        "denoised latent",
        "decoder",
        "image sample",
    ]

    plt.figure(figsize=(12, 2.2))
    ax = plt.gca()
    ax.axis("off")
    for i, text in enumerate(boxes):
        x = i / (len(boxes) - 1)
        ax.text(
            x,
            0.5,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f2f2f2", edgecolor="#555555"),
            transform=ax.transAxes,
        )
        if i < len(boxes) - 1:
            ax.annotate(
                "",
                xy=((i + 0.72) / (len(boxes) - 1), 0.5),
                xytext=((i + 0.28) / (len(boxes) - 1), 0.5),
                arrowprops=dict(arrowstyle="->", color="#333333"),
                xycoords=ax.transAxes,
                textcoords=ax.transAxes,
            )
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model, ckpt_args = load_model(Path(args.ckpt), device)

    timesteps = args.timesteps or ckpt_args.get("timesteps", 1000)
    null_label = ckpt_args.get("null_label", 10)
    schedule = DDPMSchedule(timesteps=timesteps, device=device)

    base_labels = torch.arange(10, device=device).repeat_interleave(args.num_per_class)
    samples_by_scale = []
    for scale in args.scales:
        samples = sample_ddpm_cfg(
            model=model,
            shape=(base_labels.shape[0], 1, 28, 28),
            y=base_labels,
            schedule=schedule,
            cfg_scale=scale,
            null_label=null_label,
            device=device,
        )
        samples_by_scale.append((scale, samples))

    save_cfg_comparison(
        samples_by_scale=samples_by_scale,
        save_path=args.save_path,
        num_per_class=args.num_per_class,
    )

    class_cond_samples = samples_by_scale[min(1, len(samples_by_scale) - 1)][1]
    save_image_grid(
        class_cond_samples,
        ROOT / "figures" / "day4" / "class_cond_samples.png",
        nrow=10 * args.num_per_class,
    )
    save_latent_diffusion_pipeline(ROOT / "figures" / "day4" / "latent_diffusion_pipeline.png")

    print(f"Saved CFG comparison to: {args.save_path}")


if __name__ == "__main__":
    main()
