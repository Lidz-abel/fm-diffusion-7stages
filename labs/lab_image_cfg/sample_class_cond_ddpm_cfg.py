from __future__ import annotations

from pathlib import Path
import argparse
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cfg_sampler import sample_ddim_cfg, sample_ddpm_cfg
from src.class_cond_unet import build_class_cond_unet_from_args
from src.diffusion import DDPMSchedule
from src.experiment_utils import write_json
from src.utils import save_image_grid


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
    defaults.update(config.get("sample", {}))
    defaults.update(config.get("output", {}))
    keys = {
        "ckpt",
        "in_channels",
        "image_size",
        "num_classes",
        "null_label",
        "base_channels",
        "time_dim",
        "use_attention",
        "time_scale",
        "attention_resolutions",
        "class_id",
        "class_ids",
        "all_classes",
        "num_samples",
        "num_per_class",
        "cfg_scale",
        "timesteps",
        "beta_start",
        "beta_end",
        "schedule_type",
        "cosine_s",
        "prediction_type",
        "sampler",
        "ddim_steps",
        "ddim_eta",
        "clip_x0",
        "use_ema",
        "device",
        "save_path",
        "config_out",
        "upscale",
        "padding",
        "seed",
    }
    return {key: defaults[key] for key in keys if key in defaults}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, required=False)
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--null_label", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=128)
    parser.add_argument("--time_dim", type=int, default=512)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--time_scale", type=float, default=1.0)
    parser.add_argument("--attention_resolutions", type=int, nargs="*", default=None)
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--class_ids", type=int, nargs="*", default=None)
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_per_class", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument("--schedule_type", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--cosine_s", type=float, default=0.008)
    parser.add_argument("--prediction_type", type=str, default="v_prediction", choices=["epsilon", "v_prediction"])
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--clip_x0", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "image_cfg" / "samples.png"))
    parser.add_argument("--config_out", type=str, default=str(ROOT / "results" / "image_cfg" / "sample_config.json"))
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--padding", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        parser.set_defaults(**load_config_defaults(Path(config_args.config)))
    return parser.parse_args()


def resolve_model_args(args: argparse.Namespace, ckpt_args: dict) -> dict:
    resolved = vars(args).copy()
    for key, value in ckpt_args.items():
        if value is not None:
            resolved[key] = value
    resolved["image_size"] = int(ckpt_args.get("image_size", args.image_size))
    resolved["in_channels"] = int(ckpt_args.get("in_channels", args.in_channels))
    resolved["num_classes"] = int(ckpt_args.get("num_classes", args.num_classes))
    resolved["null_label"] = int(ckpt_args.get("null_label", args.null_label or resolved["num_classes"]))
    return resolved


def load_model(args: argparse.Namespace, device: torch.device):
    if not args.ckpt:
        raise ValueError("--ckpt is required for sampling.")
    checkpoint = torch.load(args.ckpt, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model_args = resolve_model_args(args, ckpt_args)
    model = build_class_cond_unet_from_args(model_args).to(device)
    if args.use_ema and checkpoint.get("ema") is not None:
        model.load_state_dict(checkpoint["ema"]["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, model_args, ckpt_args


def build_labels(args: argparse.Namespace, num_classes: int, device: torch.device) -> tuple[torch.Tensor, int]:
    if args.all_classes:
        labels = []
        for class_id in range(num_classes):
            labels.extend([class_id] * args.num_per_class)
        return torch.tensor(labels, dtype=torch.long, device=device), args.num_per_class
    if args.class_ids:
        labels = []
        for class_id in args.class_ids:
            if not 0 <= class_id < num_classes:
                raise ValueError(f"class_id {class_id} is outside [0, {num_classes}).")
            labels.extend([class_id] * args.num_per_class)
        return torch.tensor(labels, dtype=torch.long, device=device), args.num_per_class
    if not 0 <= args.class_id < num_classes:
        raise ValueError(f"class_id {args.class_id} is outside [0, {num_classes}).")
    labels = torch.full((args.num_samples,), args.class_id, dtype=torch.long, device=device)
    return labels, max(1, int(math.sqrt(args.num_samples)))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, model_args, ckpt_args = load_model(args, device)

    timesteps = int(ckpt_args.get("timesteps", args.timesteps))
    schedule_type = ckpt_args.get("schedule_type", args.schedule_type)
    prediction_type = ckpt_args.get("prediction_type", args.prediction_type)
    schedule = DDPMSchedule(
        timesteps=timesteps,
        beta_start=float(ckpt_args.get("beta_start", args.beta_start)),
        beta_end=float(ckpt_args.get("beta_end", args.beta_end)),
        schedule_type=schedule_type,
        cosine_s=float(ckpt_args.get("cosine_s", args.cosine_s)),
        device=device,
    )

    num_classes = int(model_args["num_classes"])
    null_label = int(model_args["null_label"])
    image_size = int(model_args["image_size"])
    in_channels = int(model_args["in_channels"])
    labels, nrow = build_labels(args, num_classes, device)
    shape = (labels.shape[0], in_channels, image_size, image_size)

    if args.sampler == "ddpm":
        samples = sample_ddpm_cfg(
            model=model,
            shape=shape,
            y=labels,
            schedule=schedule,
            cfg_scale=args.cfg_scale,
            null_label=null_label,
            prediction_type=prediction_type,
            device=device,
        )
    else:
        samples = sample_ddim_cfg(
            model=model,
            shape=shape,
            y=labels,
            schedule=schedule,
            cfg_scale=args.cfg_scale,
            null_label=null_label,
            prediction_type=prediction_type,
            num_steps=args.ddim_steps,
            eta=args.ddim_eta,
            clip_x0=args.clip_x0,
            device=device,
        )

    record = vars(args).copy()
    record.update(
        {
            "resolved_model_args": model_args,
            "resolved_timesteps": timesteps,
            "resolved_schedule_type": schedule_type,
            "resolved_prediction_type": prediction_type,
            "sample_shape": list(shape),
        }
    )
    write_json(args.config_out, record)
    save_image_grid(
        samples.clamp(-1.0, 1.0),
        args.save_path,
        nrow=nrow,
        padding=args.padding,
        upscale=args.upscale,
    )
    print(f"device: {device}")
    print(f"sample shape: {shape}")
    print(f"Saved {args.sampler.upper()} samples to: {args.save_path}")
    print(f"Saved sample config to: {args.config_out}")


if __name__ == "__main__":
    main()
