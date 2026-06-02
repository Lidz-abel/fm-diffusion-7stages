from pathlib import Path
import argparse
import sys

import torch
from torch.optim import Adam
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.diffusion import DDPMSchedule, ddpm_conditional_loss
from src.dit.model import MiniDiT
from src.mnist_dataset import get_mnist_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "mnist_dit.yaml"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--null_label", type=int, default=10)
    parser.add_argument("--drop_label_prob", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_path", type=str, default=str(ROOT / "checkpoints" / "mnist_dit.pt"))
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


def save_checkpoint(model, args, loss_history, save_path: Path, step: int, epoch: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "loss": loss_history,
            "step": step,
            "epoch": epoch,
        },
        save_path,
    )


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataloader = get_mnist_dataloader(
        batch_size=args.batch_size,
        root=args.data_dir,
        num_workers=args.num_workers,
        download=True,
    )
    schedule = DDPMSchedule(timesteps=args.timesteps, device=device)
    model = MiniDiT(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        null_label=args.null_label,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    global_step = 0
    loss_history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dataloader, desc=f"MiniDiT epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            loss = ddpm_conditional_loss(
                model=model,
                x0=images,
                y=labels,
                schedule=schedule,
                null_label=args.null_label,
                drop_label_prob=args.drop_label_prob,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            loss_history.append(float(loss.item()))
            pbar.set_postfix(loss=f"{loss.item():.5f}", step=global_step)

            if global_step % args.save_every == 0:
                save_checkpoint(model, args, loss_history, save_path, global_step, epoch)
            if args.max_batches is not None and global_step >= args.max_batches:
                break

        if args.max_batches is not None and global_step >= args.max_batches:
            break

    save_checkpoint(model, args, loss_history, save_path, global_step, epoch)
    print(f"Saved MiniDiT checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
