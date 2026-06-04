from pathlib import Path
import argparse
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cfg_sampler import sample_ddim_cfg, sample_ddpm_cfg
from src.cifar10_dataset import CIFAR10_CLASSES
from src.diffusion import DDPMSchedule
from src.experiment_utils import write_json
from src.image_unet import CIFAR10FlowUNet
from src.utils import save_image_grid


def load_config_defaults(config_path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Install PyYAML or run without --config.") from exc

    with config_path.open("r") as f:
        config = yaml.safe_load(f) or {}

    defaults = {}
    defaults.update(config.get("model", {}))
    defaults.update(config.get("sample", {}))
    defaults.update(config.get("output", {}))
    keys = {
        "ckpt",
        "class_id",
        "all_classes",
        "num_samples",
        "num_per_class",
        "cfg_scale",
        "timesteps",
        "schedule_type",
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
    }
    return {key: defaults[key] for key in keys if key in defaults}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_ddpm_strong.pt"))
    parser.add_argument("--class_id", type=int, default=3)
    parser.add_argument("--all_classes", action="store_true")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_per_class", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--schedule_type", type=str, default=None, choices=["linear", "cosine"])
    parser.add_argument("--prediction_type", type=str, default=None, choices=["epsilon", "v_prediction"])
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=100)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--clip_x0", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "cifar_flow" / "final_ddpm_strong_cfg2.png"))
    parser.add_argument("--config_out", type=str, default=str(ROOT / "results" / "cifar_flow" / "sample_config_ddpm_strong.json"))
    parser.add_argument("--upscale", type=int, default=1)
    parser.add_argument("--padding", type=int, default=2)
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        parser.set_defaults(**load_config_defaults(Path(config_args.config)))
    return parser.parse_args()


def load_model(ckpt_path: Path, device: torch.device, use_ema: bool):
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint.get("args", {})
    model = CIFAR10FlowUNet(
        in_channels=3,
        base_channels=ckpt_args.get("base_channels", 192),
        emb_dim=ckpt_args.get("time_dim", 512),
        num_classes=ckpt_args.get("num_classes", 10),
        null_label=ckpt_args.get("null_label", 10),
        use_attention=ckpt_args.get("use_attention", True),
        time_scale=ckpt_args.get("time_scale", 1.0),
    ).to(device)
    if use_ema and checkpoint.get("ema") is not None:
        model.load_state_dict(checkpoint["ema"]["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ckpt_args


def build_labels(args, device: torch.device) -> torch.Tensor:
    if args.all_classes:
        labels = []
        for class_id in range(len(CIFAR10_CLASSES)):
            labels.extend([class_id] * args.num_per_class)
        return torch.tensor(labels, dtype=torch.long, device=device)
    return torch.full((args.num_samples,), args.class_id, dtype=torch.long, device=device)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(Path(args.ckpt), device, use_ema=args.use_ema)
    timesteps = args.timesteps or ckpt_args.get("timesteps", 1000)
    schedule_type = args.schedule_type or ckpt_args.get("schedule_type", "linear")
    prediction_type = args.prediction_type or ckpt_args.get("prediction_type", "epsilon")
    config_record = vars(args).copy()
    config_record.update(
        {
            "resolved_timesteps": timesteps,
            "resolved_schedule_type": schedule_type,
            "resolved_prediction_type": prediction_type,
        }
    )
    write_json(args.config_out, config_record)
    schedule = DDPMSchedule(
        timesteps=timesteps,
        beta_start=ckpt_args.get("beta_start", 1e-4),
        beta_end=ckpt_args.get("beta_end", 2e-2),
        schedule_type=schedule_type,
        cosine_s=ckpt_args.get("cosine_s", 0.008),
        device=device,
    )
    labels = build_labels(args, device)
    if args.sampler == "ddpm":
        samples = sample_ddpm_cfg(
            model=model,
            shape=(labels.shape[0], 3, 32, 32),
            y=labels,
            schedule=schedule,
            cfg_scale=args.cfg_scale,
            null_label=ckpt_args.get("null_label", 10),
            prediction_type=prediction_type,
            device=device,
        )
    else:
        samples = sample_ddim_cfg(
            model=model,
            shape=(labels.shape[0], 3, 32, 32),
            y=labels,
            schedule=schedule,
            cfg_scale=args.cfg_scale,
            null_label=ckpt_args.get("null_label", 10),
            prediction_type=prediction_type,
            num_steps=args.ddim_steps,
            eta=args.ddim_eta,
            clip_x0=args.clip_x0,
            device=device,
        )
    nrow = args.num_per_class if args.all_classes else max(1, int(math.sqrt(labels.shape[0])))
    save_image_grid(
        samples.clamp(-1.0, 1.0),
        args.save_path,
        nrow=nrow,
        padding=args.padding,
        upscale=args.upscale,
    )
    print(f"Saved {args.sampler.upper()} samples to: {args.save_path}")
    print(f"Saved sample config to: {args.config_out}")


if __name__ == "__main__":
    main()
