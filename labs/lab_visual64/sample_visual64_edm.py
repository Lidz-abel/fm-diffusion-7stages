from pathlib import Path
import argparse
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.edm import EDMSamplerConfig, sample_edm
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
        "use_ema",
        "class_id",
        "class_ids",
        "num_samples",
        "num_per_class",
        "cfg_scale",
        "sigma_data",
        "sigma_min",
        "sigma_max",
        "rho",
        "num_steps",
        "solver",
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
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "visual64_unet_edm.pt"))
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--class_id", type=int, default=0)
    parser.add_argument("--class_ids", type=int, nargs="*", default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--num_per_class", type=int, default=8)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--sigma_data", type=float, default=0.5)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "visual64" / "edm_samples.png"))
    parser.add_argument("--config_out", type=str, default=str(ROOT / "results" / "visual64" / "sample_config_edm.json"))
    parser.add_argument("--upscale", type=int, default=2)
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
        base_channels=ckpt_args.get("base_channels", 128),
        emb_dim=ckpt_args.get("time_dim", 512),
        num_classes=ckpt_args["num_classes"],
        null_label=ckpt_args["null_label"],
        use_attention=ckpt_args.get("use_attention", True),
        time_scale=ckpt_args.get("time_scale", 1.0),
    ).to(device)
    if use_ema and checkpoint.get("ema") is not None:
        model.load_state_dict(checkpoint["ema"]["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, ckpt_args


def build_labels(args, ckpt_args: dict, device: torch.device) -> tuple[torch.Tensor, int]:
    num_classes = int(ckpt_args["num_classes"])
    if args.class_ids:
        class_ids = [class_id for class_id in args.class_ids if 0 <= class_id < num_classes]
        if not class_ids:
            raise ValueError("No valid class_ids were provided.")
        labels = []
        for class_id in class_ids:
            labels.extend([class_id] * args.num_per_class)
        return torch.tensor(labels, dtype=torch.long, device=device), args.num_per_class
    return torch.full((args.num_samples,), args.class_id, dtype=torch.long, device=device), max(
        1, int(math.sqrt(args.num_samples))
    )


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(Path(args.ckpt), device, use_ema=args.use_ema)
    labels, nrow = build_labels(args, ckpt_args, device)
    sampler_config = EDMSamplerConfig(
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        num_steps=args.num_steps,
    )
    config_record = vars(args).copy()
    config_record.update({"resolved_model_args": ckpt_args})
    write_json(args.config_out, config_record)

    samples = sample_edm(
        model=model,
        shape=(labels.shape[0], 3, int(ckpt_args["image_size"]), int(ckpt_args["image_size"])),
        y=labels,
        device=device,
        config=sampler_config,
        cfg_scale=args.cfg_scale,
        null_label=ckpt_args["null_label"],
        solver=args.solver,
    )
    save_image_grid(samples, args.save_path, nrow=nrow, padding=args.padding, upscale=args.upscale)
    print(f"Saved visual64 EDM samples to: {args.save_path}")
    print(f"Saved sample config to: {args.config_out}")


if __name__ == "__main__":
    main()
