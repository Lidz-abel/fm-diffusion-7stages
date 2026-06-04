from pathlib import Path
import argparse
import os
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "cifar10_unet_edm_resume_300ep.pt"))
    parser.add_argument(
        "--log_path",
        type=str,
        default=str(ROOT / "results" / "cifar_flow" / "training_log_edm_resume_300ep.csv"),
    )
    parser.add_argument("--target_epoch", type=int, default=300)
    parser.add_argument("--check_interval", type=int, default=300)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--sample_config",
        type=str,
        default=str(ROOT / "configs" / "cifar10_edm_sampling_resume_300ep.yaml"),
    )
    parser.add_argument(
        "--metric_config",
        type=str,
        default=str(ROOT / "configs" / "cifar10_metrics_edm_resume_300ep_1k.yaml"),
    )
    parser.add_argument(
        "--formal_metric_config",
        type=str,
        default=str(ROOT / "configs" / "cifar10_metrics_edm_resume_300ep_5k.yaml"),
    )
    parser.add_argument("--run_formal_metrics", action="store_true")
    return parser.parse_args()


def read_log_epoch(log_path: Path) -> int:
    if not log_path.exists():
        return -1
    try:
        with log_path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            offset = min(end, 4096)
            f.seek(end - offset)
            lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
        for line in reversed(lines):
            parts = line.split(",")
            if len(parts) >= 2 and parts[0] != "step":
                return int(parts[1])
    except (OSError, ValueError):
        return -1
    return -1


def read_ckpt_epoch(ckpt_path: Path) -> int:
    if not ckpt_path.exists():
        return -1
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    return int(checkpoint.get("epoch", -1))


def run_command(command: list[str], gpu: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print(f"[{time.strftime('%F %T')}] run: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    log_path = Path(args.log_path)

    while True:
        log_epoch = read_log_epoch(log_path)
        print(
            f"[{time.strftime('%F %T')}] train_log={log_path} epoch={log_epoch} target={args.target_epoch}",
            flush=True,
        )
        if log_epoch >= args.target_epoch:
            ckpt_epoch = read_ckpt_epoch(ckpt_path)
            print(
                f"[{time.strftime('%F %T')}] checkpoint={ckpt_path} epoch={ckpt_epoch} target={args.target_epoch}",
                flush=True,
            )
            if ckpt_epoch >= args.target_epoch:
                break
        time.sleep(args.check_interval)

    run_command(
        [
            sys.executable,
            "labs/lab_cifar_flow/sample_cifar10_edm.py",
            "--config",
            args.sample_config,
        ],
        gpu=args.gpu,
    )
    run_command(
        [
            sys.executable,
            "labs/lab_cifar_flow/eval_cifar10_edm_metrics.py",
            "--config",
            args.metric_config,
        ],
        gpu=args.gpu,
    )
    if args.run_formal_metrics:
        run_command(
            [
                sys.executable,
                "labs/lab_cifar_flow/eval_cifar10_edm_metrics.py",
                "--config",
                args.formal_metric_config,
            ],
            gpu=args.gpu,
        )
    print(f"[{time.strftime('%F %T')}] finalization complete", flush=True)


if __name__ == "__main__":
    main()
