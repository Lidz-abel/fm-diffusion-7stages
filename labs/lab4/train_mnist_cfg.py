from pathlib import Path
import argparse
import sys

import torch
from torch.optim import Adam
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.diffusion import DDPMSchedule, ddpm_conditional_loss
from src.mnist_dataset import get_mnist_dataloader
from src.unet import MNISTConditionalUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--null_label", type=int, default=10)
    parser.add_argument("--drop_label_prob", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--save_path", type=str, default=str(ROOT / "checkpoints" / "mnist_cfg_unet.pt"))
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=None)
    return parser.parse_args()


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
    model = MNISTConditionalUNet(
        in_channels=1,
        base_channels=args.base_channels,
        emb_dim=args.time_dim,
        num_classes=args.num_classes,
        null_label=args.null_label,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    global_step = 0
    loss_history = []
    model.train()

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
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
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": global_step,
                        "epoch": epoch,
                        "loss": loss_history,
                        "args": vars(args),
                    },
                    save_path,
                )

            if args.max_batches is not None and global_step >= args.max_batches:
                break

        if args.max_batches is not None and global_step >= args.max_batches:
            break

    torch.save(
        {
            "model": model.state_dict(),
            "step": global_step,
            "epoch": epoch,
            "loss": loss_history,
            "args": vars(args),
        },
        save_path,
    )
    print(f"Saved checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
