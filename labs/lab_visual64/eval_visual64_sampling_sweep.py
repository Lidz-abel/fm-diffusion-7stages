from pathlib import Path
import argparse
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.edm import EDMSamplerConfig, sample_edm
from src.experiment_utils import write_json
from src.utils import save_image_grid, save_labeled_image_rows

from labs.lab_visual64.sample_visual64_edm import load_config_defaults, load_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--cfg_scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 3.0])
    parser.add_argument("--num_steps_list", type=int, nargs="+", default=[30, 50, 80])
    parser.add_argument("--class_ids", type=int, nargs="+", default=None)
    parser.add_argument("--num_per_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "figures" / "visual64" / "sampling_sweep"))
    parser.add_argument("--results_dir", type=str, default=str(ROOT / "results" / "visual64" / "sampling_sweep"))
    config_args, _ = parser.parse_known_args()
    defaults = load_config_defaults(Path(config_args.config))
    parser.set_defaults(**defaults)
    return parser.parse_args()


def labels_from_args(args, ckpt_args: dict, device: torch.device) -> tuple[torch.Tensor, int]:
    num_classes = int(ckpt_args["num_classes"])
    class_ids = args.class_ids
    if class_ids is None:
        class_ids = [int(args.class_id)]
    class_ids = [class_id for class_id in class_ids if 0 <= class_id < num_classes]
    if not class_ids:
        raise ValueError("No valid class ids.")
    labels = []
    for class_id in class_ids:
        labels.extend([class_id] * args.num_per_class)
    return torch.tensor(labels, dtype=torch.long, device=device), args.num_per_class


def main():
    args = parse_args()
    device_name = args.device if args.device is not None else "cuda"
    device = torch.device(device_name if device_name != "cuda" or torch.cuda.is_available() else "cpu")
    model, ckpt_args = load_model(Path(args.ckpt), device, use_ema=args.use_ema)
    labels, nrow = labels_from_args(args, ckpt_args, device)
    image_size = int(ckpt_args["image_size"])

    output_dir = Path(args.output_dir) / args.tag
    results_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    row_tensors = []
    row_labels = []
    run_records = []
    col_labels = [f"c{int(label)}" for label in labels.detach().cpu().tolist()]

    for num_steps in args.num_steps_list:
        for cfg_scale in args.cfg_scales:
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)
            sampler_config = EDMSamplerConfig(
                sigma_data=args.sigma_data,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                rho=args.rho,
                num_steps=num_steps,
            )
            samples = sample_edm(
                model=model,
                shape=(labels.shape[0], 3, image_size, image_size),
                y=labels,
                device=device,
                config=sampler_config,
                cfg_scale=cfg_scale,
                null_label=ckpt_args["null_label"],
                solver=args.solver,
            )
            name = f"{args.tag}_steps{num_steps}_cfg{cfg_scale:g}.png"
            save_path = output_dir / name
            save_image_grid(samples, save_path, nrow=nrow, padding=args.padding, upscale=args.upscale)

            row_tensors.append(samples.detach().cpu())
            row_labels.append(f"steps={num_steps}, cfg={cfg_scale:g}")
            run_records.append(
                {
                    "tag": args.tag,
                    "save_path": str(save_path),
                    "num_steps": num_steps,
                    "cfg_scale": cfg_scale,
                    "solver": args.solver,
                    "seed": args.seed,
                    "class_ids": args.class_ids,
                    "num_per_class": args.num_per_class,
                    "checkpoint": args.ckpt,
                    "use_ema": args.use_ema,
                }
            )
            print(f"Saved {save_path}")

    panel_path = output_dir / f"{args.tag}_sampling_sweep_panel.png"
    save_labeled_image_rows(
        image_rows=row_tensors,
        row_labels=row_labels,
        col_labels=col_labels,
        save_path=panel_path,
        title=f"Visual64 EDM sampling sweep: {args.tag}",
    )
    write_json(results_dir / f"{args.tag}_sampling_sweep.json", {"args": vars(args), "runs": run_records})
    print(f"Saved panel to: {panel_path}")
    print(f"Saved records to: {results_dir / f'{args.tag}_sampling_sweep.json'}")


if __name__ == "__main__":
    main()
