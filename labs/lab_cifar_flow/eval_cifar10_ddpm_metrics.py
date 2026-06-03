from pathlib import Path
import argparse
import math
import sys
import time

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.cfg_sampler import sample_ddim_cfg, sample_ddpm_cfg
from src.cifar10_dataset import get_cifar10_dataloader
from src.diffusion import DDPMSchedule
from src.experiment_utils import append_csv_row, set_seed, write_json
from src.image_metrics import InceptionFeatureExtractor, compute_stats, frechet_distance, inception_score
from src.utils import save_image_grid

from labs.lab_cifar_flow.sample_cifar10_ddpm import load_model


def stats_to_device(stats, device: torch.device):
    stats.mean = stats.mean.to(device)
    stats.covariance = stats.covariance.to(device)
    if stats.logits is not None:
        stats.logits = stats.logits.to(device)
    return stats


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
    defaults.update(config.get("metrics", {}))
    defaults.update(config.get("output", {}))
    keys = {
        "ckpt",
        "use_ema",
        "sampler",
        "timesteps",
        "schedule_type",
        "prediction_type",
        "ddim_steps",
        "ddim_eta",
        "clip_x0",
        "cfg_scale",
        "num_samples",
        "batch_size",
        "real_batch_size",
        "device",
        "data_dir",
        "real_split",
        "num_workers",
        "pretrained_inception",
        "inception_splits",
        "seed",
        "save_grid",
        "json_path",
        "csv_path",
    }
    return {key: defaults[key] for key in keys if key in defaults}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--sampler", type=str, default="ddim", choices=["ddpm", "ddim"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--schedule_type", type=str, default=None, choices=["linear", "cosine"])
    parser.add_argument("--prediction_type", type=str, default=None, choices=["epsilon", "v_prediction"])
    parser.add_argument("--ddim_steps", type=int, default=250)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--clip_x0", action="store_true")
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--real_batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default=str(ROOT / "data"))
    parser.add_argument("--real_split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pretrained_inception", action="store_true")
    parser.add_argument("--inception_splits", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_grid", type=str, default=str(ROOT / "figures" / "cifar_flow" / "metrics_eval_samples.png"))
    parser.add_argument("--json_path", type=str, default=str(ROOT / "results" / "cifar_flow" / "metrics_ddpm.json"))
    parser.add_argument("--csv_path", type=str, default=str(ROOT / "results" / "cifar_flow" / "metrics_ddpm.csv"))
    config_args, _ = parser.parse_known_args()
    if config_args.config is not None:
        parser.set_defaults(**load_config_defaults(Path(config_args.config)))
    return parser.parse_args()


@torch.no_grad()
def collect_real_features(
    extractor: InceptionFeatureExtractor,
    num_samples: int,
    batch_size: int,
    device: torch.device,
    data_dir: str,
    real_split: str,
    num_workers: int,
) -> torch.Tensor:
    train = real_split == "train"
    dataloader = get_cifar10_dataloader(
        batch_size=batch_size,
        train=train,
        root=data_dir,
        num_workers=num_workers,
        download=True,
        augment=False,
        shuffle=True,
    )
    features = []
    seen = 0
    pbar = tqdm(total=num_samples, desc="Real CIFAR features")
    for images, _ in dataloader:
        images = images.to(device)
        feat, _ = extractor(images, input_range="minus_one_one")
        features.append(feat.cpu())
        seen += images.shape[0]
        pbar.update(min(images.shape[0], max(num_samples - (seen - images.shape[0]), 0)))
        if seen >= num_samples:
            break
    pbar.close()
    return torch.cat(features, dim=0)[:num_samples]


@torch.no_grad()
def collect_fake_features(
    model,
    schedule: DDPMSchedule,
    extractor: InceptionFeatureExtractor,
    args,
    ckpt_args: dict,
    device: torch.device,
    prediction_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    all_features = []
    all_logits = []
    preview = None
    generated = 0
    num_classes = ckpt_args.get("num_classes", 10)
    null_label = ckpt_args.get("null_label", 10)

    pbar = tqdm(total=args.num_samples, desc="Fake sample features")
    while generated < args.num_samples:
        batch = min(args.batch_size, args.num_samples - generated)
        labels = torch.arange(generated, generated + batch, device=device) % num_classes
        shape = (batch, 3, 32, 32)
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
        samples = samples.clamp(-1.0, 1.0)
        if preview is None:
            preview = samples[: min(samples.shape[0], 64)].detach().cpu()
        feat, logits = extractor(samples, input_range="minus_one_one")
        all_features.append(feat.cpu())
        all_logits.append(logits.cpu())
        generated += batch
        pbar.update(batch)
    pbar.close()
    return torch.cat(all_features, dim=0), torch.cat(all_logits, dim=0), preview


def main():
    args = parse_args()
    if args.ckpt is None:
        raise ValueError("Provide --ckpt or set model.ckpt in --config.")
    set_seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(Path(args.ckpt), device, use_ema=args.use_ema)
    timesteps = args.timesteps or ckpt_args.get("timesteps", 1000)
    schedule_type = args.schedule_type or ckpt_args.get("schedule_type", "linear")
    prediction_type = args.prediction_type or ckpt_args.get("prediction_type", "epsilon")
    schedule = DDPMSchedule(
        timesteps=timesteps,
        beta_start=ckpt_args.get("beta_start", 1e-4),
        beta_end=ckpt_args.get("beta_end", 2e-2),
        schedule_type=schedule_type,
        cosine_s=ckpt_args.get("cosine_s", 0.008),
        device=device,
    )
    extractor = InceptionFeatureExtractor(pretrained=args.pretrained_inception).to(device)

    started = time.time()
    real_features = collect_real_features(
        extractor=extractor,
        num_samples=args.num_samples,
        batch_size=args.real_batch_size,
        device=device,
        data_dir=args.data_dir,
        real_split=args.real_split,
        num_workers=args.num_workers,
    )
    fake_features, fake_logits, preview = collect_fake_features(
        model=model,
        schedule=schedule,
        extractor=extractor,
        args=args,
        ckpt_args=ckpt_args,
        device=device,
        prediction_type=prediction_type,
    )
    real_stats = compute_stats(real_features)
    fake_stats = compute_stats(fake_features, fake_logits)
    real_stats = stats_to_device(real_stats, device)
    fake_stats = stats_to_device(fake_stats, device)
    fid = frechet_distance(real_stats, fake_stats)
    is_mean, is_std = inception_score(fake_logits.to(device), splits=args.inception_splits)
    elapsed = time.time() - started

    if preview is not None and args.save_grid:
        nrow = max(1, int(math.sqrt(preview.shape[0])))
        save_image_grid(preview, args.save_grid, nrow=nrow)

    result = {
        "checkpoint": args.ckpt,
        "use_ema": args.use_ema,
        "sampler": args.sampler,
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "cfg_scale": args.cfg_scale,
        "timesteps": timesteps,
        "schedule_type": schedule_type,
        "prediction_type": prediction_type,
        "ddim_steps": args.ddim_steps if args.sampler == "ddim" else None,
        "ddim_eta": args.ddim_eta if args.sampler == "ddim" else None,
        "clip_x0": args.clip_x0 if args.sampler == "ddim" else None,
        "pretrained_inception": args.pretrained_inception,
        "real_split": args.real_split,
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "seconds_total": elapsed,
        "seconds_per_sample": elapsed / max(args.num_samples, 1),
        "seed": args.seed,
    }
    write_json(args.json_path, result)
    append_csv_row(args.csv_path, result)
    print(result)


if __name__ == "__main__":
    main()
