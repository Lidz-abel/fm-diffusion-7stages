from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from labs.lab_cifar_flow.sample_cifar10_fm import load_unet
from src.benchmark_utils import time_sampling
from src.experiment_utils import append_csv_row, count_parameters, set_seed
from src.image_flow_samplers import sample_flow


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_fm.pt"))
    parser.add_argument("--model_type", type=str, default="unet")
    parser.add_argument("--solvers", type=str, nargs="+", default=["euler", "heun"])
    parser.add_argument("--nfe_list", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--num_samples", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runtime_csv", type=str, default=str(ROOT / "results" / "cifar_flow" / "runtime_summary.csv"))
    parser.add_argument("--memory_csv", type=str, default=str(ROOT / "results" / "cifar_flow" / "memory_summary.csv"))
    parser.add_argument("--figure_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "runtime_vs_nfe.png"))
    return parser.parse_args()


def save_runtime_plot(rows: list[dict], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for solver in sorted({row["solver"] for row in rows}):
        solver_rows = [row for row in rows if row["solver"] == solver]
        x = [int(row["nfe"]) for row in solver_rows]
        y = [float(row["seconds_per_image"]) for row in solver_rows]
        plt.plot(x, y, marker="o", label=solver)
    plt.xlabel("NFE")
    plt.ylabel("seconds / image")
    plt.title("CIFAR-10 Rectified Flow Sampling Runtime")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_unet(Path(args.ckpt), device)
    labels = (torch.arange(args.num_samples, device=device) % 10).long()
    Path(args.runtime_csv).unlink(missing_ok=True)
    Path(args.memory_csv).unlink(missing_ok=True)
    rows = []
    param_count = count_parameters(model)

    for solver in args.solvers:
        for nfe in args.nfe_list:
            _, timing = time_sampling(
                lambda solver=solver, nfe=nfe: sample_flow(
                    model=model,
                    shape=(args.num_samples, 3, 32, 32),
                    y=labels,
                    solver=solver,
                    nfe=nfe,
                    cfg_scale=args.cfg_scale,
                    null_label=ckpt_args.get("null_label", 10),
                    device=device,
                ),
                device,
            )
            row = {
                "model_type": args.model_type,
                "solver": solver,
                "nfe": nfe,
                "cfg_scale": args.cfg_scale,
                "num_samples": args.num_samples,
                "batch_size": args.batch_size,
                "seconds_total": f"{timing.seconds_total:.6f}",
                "seconds_per_image": f"{timing.seconds_total / args.num_samples:.6f}",
                "max_memory_mb": f"{timing.max_memory_mb:.2f}",
                "device": str(device),
                "seed": args.seed,
                "checkpoint": args.ckpt,
                "parameter_count": param_count,
            }
            rows.append(row)
            append_csv_row(args.runtime_csv, row)
            append_csv_row(args.memory_csv, row)

    save_runtime_plot(rows, args.figure_path)
    print(f"Saved runtime CSV to: {args.runtime_csv}")
    print(f"Saved memory CSV to: {args.memory_csv}")
    print(f"Saved runtime figure to: {args.figure_path}")


if __name__ == "__main__":
    main()
