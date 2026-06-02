from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from labs.lab_cifar_flow.sample_cifar10_fm import load_unet
from src.benchmark_utils import time_sampling
from src.experiment_utils import append_csv_row, set_seed
from src.image_flow_samplers import sample_flow
from src.metrics_utils import tensor_summary
from src.utils import save_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_fm.pt"))
    parser.add_argument("--model_type", type=str, default="unet")
    parser.add_argument("--solvers", type=str, nargs="+", default=["euler", "heun"])
    parser.add_argument("--nfe", type=int, default=20)
    parser.add_argument("--num_per_class", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--figure_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "solver_ablation_unet.png"))
    parser.add_argument("--csv_path", type=str, default=str(ROOT / "results" / "cifar_flow" / "solver_ablation.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_unet(Path(args.ckpt), device)
    labels = torch.arange(10, device=device).repeat_interleave(args.num_per_class)
    rows = []

    for solver in args.solvers:
        samples, timing = time_sampling(
            lambda solver=solver: sample_flow(
                model=model,
                shape=(labels.shape[0], 3, 32, 32),
                y=labels,
                solver=solver,
                nfe=args.nfe,
                cfg_scale=args.cfg_scale,
                null_label=ckpt_args.get("null_label", 10),
                device=device,
            ),
            device,
        )
        rows.append(samples)
        row = {
            "model_type": args.model_type,
            "solver": solver,
            "nfe": args.nfe,
            "cfg_scale": args.cfg_scale,
            "num_samples": labels.shape[0],
            "seconds_total": f"{timing.seconds_total:.6f}",
            "seconds_per_image": f"{timing.seconds_total / labels.shape[0]:.6f}",
            "max_memory_mb": f"{timing.max_memory_mb:.2f}",
            "checkpoint": args.ckpt,
            **tensor_summary(samples),
        }
        append_csv_row(args.csv_path, row)

    save_image_grid(torch.cat(rows, dim=0), args.figure_path, nrow=labels.shape[0])
    print(f"Saved solver ablation figure to: {args.figure_path}")
    print(f"Saved solver ablation CSV to: {args.csv_path}")


if __name__ == "__main__":
    main()
