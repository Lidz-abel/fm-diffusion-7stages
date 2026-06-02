from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cifar10_dataset import get_cifar10_dataloader
from src.experiment_utils import count_parameters, resolve_device, set_seed, write_json
from src.image_flow_matching import flow_matching_loss
from src.image_unet import CIFAR10FlowUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--null_label", type=int, default=10)
    parser.add_argument("--drop_label_prob", type=float, default=0.1)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--save_path", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_fm.pt"))
    parser.add_argument("--log_path", type=str, default=str(ROOT / "results" / "cifar_flow" / "training_log.csv"))
    parser.add_argument("--config_path", type=str, default=str(ROOT / "results" / "cifar_flow" / "config_used.json"))
    parser.add_argument("--curve_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "first_stage_training_curve.png"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    return parser.parse_args()


def save_training_curve(loss_history: list[float], save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, linewidth=1.2)
    plt.xlabel("step")
    plt.ylabel("Flow Matching MSE")
    plt.title("CIFAR-10 U-Net Rectified Flow Training")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def write_log_header(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write("step,epoch,loss,lr\n")


def append_log(log_path: Path, step: int, epoch: int, loss: float, lr: float) -> None:
    with log_path.open("a") as f:
        f.write(f"{step},{epoch},{loss:.8f},{lr:.8g}\n")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path)
    write_log_header(log_path)
    write_json(Path(args.config_path), vars(args))

    dataloader = get_cifar10_dataloader(
        batch_size=args.batch_size,
        train=True,
        root=args.data_dir,
        num_workers=args.num_workers,
        download=True,
    )

    model = CIFAR10FlowUNet(
        in_channels=3,
        base_channels=args.base_channels,
        emb_dim=args.time_dim,
        num_classes=args.num_classes,
        null_label=args.null_label,
        use_attention=args.use_attention,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    loss_history: list[float] = []
    model.train()

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            loss = flow_matching_loss(
                model=model,
                x1=images,
                y=labels,
                null_label=args.null_label,
                drop_label_prob=args.drop_label_prob,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            global_step += 1
            loss_value = float(loss.item())
            loss_history.append(loss_value)
            append_log(log_path, global_step, epoch, loss_value, args.lr)
            pbar.set_postfix(loss=f"{loss_value:.5f}", step=global_step)

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

            if args.max_steps is not None and global_step >= args.max_steps:
                break
        if args.max_steps is not None and global_step >= args.max_steps:
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
    save_training_curve(loss_history, args.curve_path)
    print(f"Saved checkpoint to: {save_path}")
    print(f"Saved training log to: {log_path}")
    print(f"Saved training curve to: {args.curve_path}")


if __name__ == "__main__":
    main()
