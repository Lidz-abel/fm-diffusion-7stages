from pathlib import Path
import argparse
import math
import sys

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cifar10_dataset import get_cifar10_dataloader
from src.ema import EMA
from src.experiment_utils import count_parameters, resolve_device, set_seed, write_json
from src.image_flow_matching import flow_matching_loss
from src.image_unet import CIFAR10FlowUNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
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
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        parser.set_defaults(**load_config_defaults(Path(config_args.config)))
    return parser.parse_args()


def load_config_defaults(config_path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML or run without --config.") from exc

    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    defaults = {}
    defaults.update(config.get("model", {}))
    defaults.update(config.get("flow", {}))
    defaults.update(config.get("train", {}))
    defaults.update(config.get("paths", {}))

    aliases = {
        "base_channels": "base_channels",
        "time_dim": "time_dim",
        "num_classes": "num_classes",
        "null_label": "null_label",
        "drop_label_prob": "drop_label_prob",
        "epochs": "epochs",
        "batch_size": "batch_size",
        "lr": "lr",
        "weight_decay": "weight_decay",
        "grad_clip": "grad_clip",
        "optimizer": "optimizer",
        "use_amp": "use_amp",
        "use_ema": "use_ema",
        "ema_decay": "ema_decay",
        "warmup_steps": "warmup_steps",
        "scheduler": "scheduler",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "device": "device",
        "num_workers": "num_workers",
        "save_every": "save_every",
        "save_path": "save_path",
        "log_path": "log_path",
        "config_path": "config_path",
        "curve_path": "curve_path",
        "seed": "seed",
        "use_attention": "use_attention",
    }
    parsed = {}
    for source_key, target_key in aliases.items():
        if source_key in defaults:
            parsed[target_key] = defaults[source_key]
    if "attention_resolutions" in defaults:
        parsed["use_attention"] = bool(defaults["attention_resolutions"])
    return parsed


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


def build_scheduler(optimizer: torch.optim.Optimizer, args, total_update_steps: int):
    if args.scheduler == "none" and args.warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return max((step + 1) / args.warmup_steps, 1e-8)
        if args.scheduler == "cosine":
            denom = max(total_update_steps - args.warmup_steps, 1)
            progress = min(max((step - args.warmup_steps) / denom, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(progress * math.pi))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


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
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_batches = len(dataloader) * args.epochs
    total_update_steps = max(total_batches // max(args.gradient_accumulation_steps, 1), 1)
    scheduler = build_scheduler(optimizer, args, total_update_steps)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    global_step = 0
    update_step = 0
    loss_history: list[float] = []
    model.train()

    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"EMA: {'enabled' if ema is not None else 'disabled'}")
    print(f"AMP: {'enabled' if scaler.is_enabled() else 'disabled'}")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (images, labels) in enumerate(pbar, start=1):
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(
                "cuda",
                enabled=args.use_amp and device.type == "cuda",
            ):
                loss = flow_matching_loss(
                    model=model,
                    x1=images,
                    y=labels,
                    null_label=args.null_label,
                    drop_label_prob=args.drop_label_prob,
                )
                scaled_loss = loss / max(args.gradient_accumulation_steps, 1)

            global_step += 1
            scaler.scale(scaled_loss).backward()

            should_update = (
                batch_idx % args.gradient_accumulation_steps == 0
                or batch_idx == len(dataloader)
                or (args.max_steps is not None and global_step >= args.max_steps)
            )
            if should_update:
                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update(model)

            loss_value = float(loss.item())
            loss_history.append(loss_value)
            current_lr = optimizer.param_groups[0]["lr"]
            append_log(log_path, global_step, epoch, loss_value, current_lr)
            pbar.set_postfix(loss=f"{loss_value:.5f}", step=global_step, updates=update_step)

            if global_step % args.save_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "ema": ema.state_dict() if ema is not None else None,
                        "step": global_step,
                        "update_step": update_step,
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
            "ema": ema.state_dict() if ema is not None else None,
            "step": global_step,
            "update_step": update_step,
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
