from pathlib import Path
import argparse
import sys

import torch
from torch.optim import Adam
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.toy_data import sample_gaussian_mixture, sample_standard_normal
from src.mlp import FlowMLP
from src.flow_matching import flow_matching_loss
from src.samplers import sample_ode_euler, sample_ode_euler_trajectory
from src.visualization import (
    plot_points,
    plot_trajectories,
    plot_nfe_samples,
    plot_vector_field,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_every", type=int, default=1000)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    figures_dir = ROOT / "figures" / "stage2"
    ckpt_dir = ROOT / "checkpoints"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = FlowMLP(
        data_dim=2,
        hidden_dim=args.hidden_dim,
        time_emb_dim=64,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    loss_history = []

    pbar = tqdm(range(1, args.train_steps + 1), desc="Training 2D Flow Matching")

    for step in pbar:
        data = sample_gaussian_mixture(args.batch_size, device=device)

        loss = flow_matching_loss(model, data)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))
        pbar.set_postfix(loss=f"{loss.item():.6f}")

        if step % args.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "step": step,
                    "loss": loss_history,
                    "args": vars(args),
                },
                ckpt_dir / "stage2_flow_matching.pt",
            )

    torch.save(
        {
            "model": model.state_dict(),
            "step": args.train_steps,
            "loss": loss_history,
            "args": vars(args),
        },
        ckpt_dir / "stage2_flow_matching.pt",
    )

    # 1. Plot real data
    real_data = sample_gaussian_mixture(5000, device=device)
    plot_points(
        real_data,
        title="Target data distribution",
        save_path=str(figures_dir / "target_data.png"),
    )

    # 2. Plot initial Gaussian noise
    noise = sample_standard_normal(5000, dim=2, device=device)
    plot_points(
        noise,
        title="Initial Gaussian noise",
        save_path=str(figures_dir / "initial_noise.png"),
    )

    # 3. NFE comparison
    model.eval()
    samples_by_nfe = {}

    for nfe in [5, 10, 20, 50, 100]:
        x0 = sample_standard_normal(5000, dim=2, device=device)
        samples = sample_ode_euler(model, x0, n_steps=nfe)
        samples_by_nfe[nfe] = samples

        plot_points(
            samples,
            title=f"Flow Matching samples, NFE={nfe}",
            save_path=str(figures_dir / f"fm_samples_nfe_{nfe}.png"),
        )

    plot_nfe_samples(
        samples_by_nfe,
        save_path=str(figures_dir / "fm_samples_nfe_compare.png"),
    )

    # 4. Trajectory visualization
    x0_traj = sample_standard_normal(100, dim=2, device=device)
    traj = sample_ode_euler_trajectory(model, x0_traj, n_steps=100)

    plot_trajectories(
        traj,
        title="Learned ODE trajectories",
        save_path=str(figures_dir / "fm_trajectories.png"),
        max_paths=80,
    )

    # 5. Vector field visualization
    for t_value in [0.0, 0.25, 0.5, 0.75]:
        plot_vector_field(
            model,
            t_value=t_value,
            save_path=str(figures_dir / f"fm_vector_field_t_{t_value:.2f}.png"),
            device=str(device),
        )

    print(f"Saved figures to: {figures_dir}")
    print(f"Saved checkpoint to: {ckpt_dir / 'stage2_flow_matching.pt'}")


if __name__ == "__main__":
    main()
