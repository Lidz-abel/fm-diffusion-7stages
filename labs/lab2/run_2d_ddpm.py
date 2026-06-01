from pathlib import Path
import argparse
import sys

import torch
from torch.optim import Adam
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.diffusion import DDPMSchedule, ddpm_loss, q_sample, sample_ddpm, sample_ddpm_trajectory
from src.mlp import FlowMLP
from src.samplers import sample_ode_euler
from src.toy_data import sample_gaussian_mixture, sample_standard_normal
from src.visualization import plot_points, plot_sample_panels


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--n_eval_samples", type=int, default=5000)

    return parser.parse_args()


def save_checkpoint(model, args, loss_history, ckpt_path: Path, step: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "loss": loss_history,
            "args": vars(args),
        },
        ckpt_path,
    )


def plot_forward_noising(
    schedule: DDPMSchedule,
    figures_dir: Path,
    device: torch.device,
    n_samples: int,
) -> None:
    data = sample_gaussian_mixture(n_samples, device=device)
    noise = torch.randn_like(data)

    indices = [
        0,
        schedule.timesteps // 4,
        schedule.timesteps // 2,
        (3 * schedule.timesteps) // 4,
        schedule.timesteps - 1,
    ]

    panels = {"t=0 data": data}
    for idx in indices[1:]:
        t = torch.full((n_samples,), idx, device=device, dtype=torch.long)
        xt = q_sample(x0=data, t=t, noise=noise, schedule=schedule)
        panels[f"t={idx}"] = xt

    plot_sample_panels(
        panels,
        save_path=str(figures_dir / "forward_noising.png"),
    )


def maybe_sample_flow_matching(
    device: torch.device,
    n_samples: int,
) -> torch.Tensor | None:
    ckpt_path = ROOT / "checkpoints" / "stage2_flow_matching.pt"
    if not ckpt_path.exists():
        return None

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

    x0 = sample_standard_normal(n_samples, dim=2, device=device)
    return sample_ode_euler(model, x0=x0, n_steps=100)


def main():
    args = parse_args()

    device = torch.device(args.device)
    figures_dir = ROOT / "figures" / "stage3"
    ckpt_dir = ROOT / "checkpoints"
    figures_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    schedule = DDPMSchedule(timesteps=args.timesteps, device=device)

    model = FlowMLP(
        data_dim=2,
        hidden_dim=args.hidden_dim,
        time_emb_dim=64,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_history = []
    ckpt_path = ckpt_dir / "stage3_ddpm.pt"

    pbar = tqdm(range(1, args.train_steps + 1), desc="Training 2D DDPM")
    for step in pbar:
        data = sample_gaussian_mixture(args.batch_size, device=device)
        loss = ddpm_loss(model=model, x0=data, schedule=schedule)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_history.append(float(loss.item()))
        pbar.set_postfix(loss=f"{loss.item():.6f}")

        if step % args.save_every == 0:
            save_checkpoint(model, args, loss_history, ckpt_path, step)

    save_checkpoint(model, args, loss_history, ckpt_path, args.train_steps)

    model.eval()

    plot_forward_noising(
        schedule=schedule,
        figures_dir=figures_dir,
        device=device,
        n_samples=args.n_eval_samples,
    )

    ddpm_samples = sample_ddpm(
        model=model,
        n_samples=args.n_eval_samples,
        schedule=schedule,
        device=device,
        dim=2,
    )
    plot_points(
        ddpm_samples,
        title="DDPM samples",
        save_path=str(figures_dir / "ddpm_samples.png"),
    )

    trajectory = sample_ddpm_trajectory(
        model=model,
        n_samples=args.n_eval_samples,
        schedule=schedule,
        device=device,
        dim=2,
        save_every=max(args.timesteps // 4, 1),
    )
    trajectory_panels = {
        f"step={step}": samples for step, samples in sorted(trajectory.items(), reverse=True)
    }
    plot_sample_panels(
        trajectory_panels,
        save_path=str(figures_dir / "ddpm_sampling_trajectory.png"),
    )

    target_data = sample_gaussian_mixture(args.n_eval_samples, device=device)
    comparison = {
        "target data": target_data,
        "DDPM samples": ddpm_samples,
    }

    fm_samples = maybe_sample_flow_matching(device=device, n_samples=args.n_eval_samples)
    if fm_samples is not None:
        comparison = {
            "target data": target_data,
            "Flow Matching samples": fm_samples,
            "DDPM samples": ddpm_samples,
        }

    plot_sample_panels(
        comparison,
        save_path=str(figures_dir / "fm_vs_ddpm.png"),
    )

    print(f"Saved figures to: {figures_dir}")
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
