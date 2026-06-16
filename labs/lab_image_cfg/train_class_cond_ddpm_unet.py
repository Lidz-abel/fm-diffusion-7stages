from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.class_cond_unet import ClassConditionalUNet
from src.diffusion import DDPMSchedule, ddpm_conditional_loss
from src.ema import EMA
from src.experiment_utils import count_parameters, resolve_device, set_seed, write_json
from src.image_dataset import get_image_dataloader


def load_config_defaults(config_path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML or run without --config.") from exc

    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    defaults = {}
    defaults.update(config.get("data", {}))
    defaults.update(config.get("model", {}))
    defaults.update(config.get("diffusion", {}))
    defaults.update(config.get("train", {}))
    defaults.update(config.get("paths", {}))
    keys = {
        "dataset",
        "data_dir",
        "image_size",
        "in_channels",
        "num_classes",
        "base_channels",
        "time_dim",
        "null_label",
        "use_attention",
        "attention_resolutions",
        "time_scale",
        "timesteps",
        "beta_start",
        "beta_end",
        "schedule_type",
        "cosine_s",
        "prediction_type",
        "drop_label_prob",
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "grad_clip",
        "use_amp",
        "use_ema",
        "ema_decay",
        "warmup_steps",
        "scheduler",
        "gradient_accumulation_steps",
        "device",
        "num_workers",
        "download",
        "synthetic_samples",
        "save_every",
        "save_path",
        "log_path",
        "config_path",
        "seed",
    }
    return {key: defaults[key] for key in keys if key in defaults}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="synthetic")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument("--null_label", type=int, default=None)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--attention_resolutions", type=int, nargs="*", default=[32, 16])
    parser.add_argument("--time_scale", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--schedule_type", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--prediction_type", type=str, default="v_prediction", choices=["epsilon", "v_prediction"])
    parser.add_argument("--drop_label_prob", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--synthetic_samples", type=int, default=16)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default=str(ROOT / "checkpoints" / "class_cond_ddpm_smoke.pt"))
    parser.add_argument("--log_path", type=str, default=str(ROOT / "results" / "image_cfg" / "training_log_smoke.csv"))
    parser.add_argument("--config_path", type=str, default=str(ROOT / "results" / "image_cfg" / "config_used_smoke.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", type=int, default=None)
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        parser.set_defaults(**load_config_defaults(Path(config_args.config)))
    return parser.parse_args()


def build_scheduler(optimizer: torch.optim.Optimizer, args: argparse.Namespace, total_updates: int):
    if args.scheduler == "none" and args.warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        if args.warmup_steps > 0 and step < args.warmup_steps:
            return max((step + 1) / args.warmup_steps, 1e-8)
        if args.scheduler == "cosine":
            denom = max(total_updates - args.warmup_steps, 1)
            progress = min(max((step - args.warmup_steps) / denom, 0.0), 1.0)
            return 0.5 * (1.0 + math.cos(progress * math.pi))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def main() -> None:
    args = parse_args()
    if args.null_label is None:
        args.null_label = args.num_classes
    set_seed(args.seed)
    device = resolve_device(args.device)

    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.config_path, vars(args))
    with Path(args.log_path).open("w") as f:
        f.write("step,epoch,loss,lr\n")

    dataloader, info = get_image_dataloader(
        dataset=args.dataset,
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        train=True,
        num_workers=args.num_workers,
        download=args.download,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        synthetic_samples=args.synthetic_samples,
    )
    if info.num_classes != args.num_classes and args.dataset != "synthetic":
        print(f"Updating num_classes from config {args.num_classes} to dataset value {info.num_classes}")
        args.num_classes = info.num_classes
        args.null_label = args.num_classes

    schedule = DDPMSchedule(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule_type=args.schedule_type,
        cosine_s=args.cosine_s,
        device=device,
    )
    model = ClassConditionalUNet(
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        time_dim=args.time_dim,
        num_classes=args.num_classes,
        null_label=args.null_label,
        use_attention=args.use_attention,
        time_scale=args.time_scale,
        image_size=args.image_size,
        attention_resolutions=args.attention_resolutions,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_batches = len(dataloader) * args.epochs
    total_updates = max(total_batches // max(args.gradient_accumulation_steps, 1), 1)
    scheduler = build_scheduler(optimizer, args, total_updates)
    ema = EMA(model, decay=args.ema_decay) if args.use_ema else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp and device.type == "cuda")

    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}, classes: {args.num_classes}, image_size: {args.image_size}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    global_step = 0
    update_step = 0
    loss_history: list[float] = []
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, (images, labels) in enumerate(pbar, start=1):
            images = images.to(device)
            labels = labels.to(device)
            with torch.amp.autocast("cuda", enabled=args.use_amp and device.type == "cuda"):
                loss = ddpm_conditional_loss(
                    model=model,
                    x0=images,
                    y=labels,
                    schedule=schedule,
                    null_label=args.null_label,
                    drop_label_prob=args.drop_label_prob,
                    prediction_type=args.prediction_type,
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
            lr = optimizer.param_groups[0]["lr"]
            with Path(args.log_path).open("a") as f:
                f.write(f"{global_step},{epoch},{loss_value:.8f},{lr:.8g}\n")
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
                    args.save_path,
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
        args.save_path,
    )
    print(f"Saved checkpoint to: {args.save_path}")


if __name__ == "__main__":
    main()
