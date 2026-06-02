from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.fast_sampling import run_nfe_ablation
from src.mlp import FlowMLP
from src.toy_data import sample_standard_normal
from src.visualization import plot_nfe_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "stage2_flow_matching.pt"))
    parser.add_argument("--nfe", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--num_samples", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "day5" / "nfe_compare.png"))
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}. Run Stage 2 training first or pass --ckpt."
        )

    device = torch.device(args.device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})

    model = FlowMLP(
        data_dim=2,
        hidden_dim=ckpt_args.get("hidden_dim", 128),
        time_emb_dim=64,
        num_layers=ckpt_args.get("num_layers", 4),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    x_init = sample_standard_normal(args.num_samples, dim=2, device=device)
    samples_by_nfe = run_nfe_ablation(model=model, x_init=x_init, nfe_list=args.nfe)
    plot_nfe_samples(samples_by_nfe, save_path=args.save_path)
    print(f"Saved NFE comparison to: {args.save_path}")


if __name__ == "__main__":
    main()
