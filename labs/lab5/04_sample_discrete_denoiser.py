from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.discrete_denoiser import DiscreteDenoiser


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "checkpoints" / "discrete_denoiser_toy.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "figures" / "day5" / "discrete_reverse_process.png"))
    return parser.parse_args()


def format_sequence(seq: torch.Tensor) -> str:
    return "[" + ", ".join("MASK" if int(x) == 0 else str(int(x)) for x in seq) + "]"


@torch.no_grad()
def sample_discrete(
    model: DiscreteDenoiser,
    seq_len: int,
    num_steps: int,
    mask_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    model.eval()
    xt = torch.full((1, seq_len), fill_value=mask_token_id, dtype=torch.long, device=device)
    trajectory = [(num_steps, xt[0].clone())]

    for step in reversed(range(num_steps)):
        t = torch.full((1,), step, dtype=torch.long, device=device)
        logits = model(xt, t)
        mask_positions = xt == mask_token_id

        if mask_positions.any():
            probs = F.softmax(logits, dim=-1)
            confidence, pred = probs.max(dim=-1)

            remaining_target = int(round(seq_len * step / max(num_steps - 1, 1)))
            current_masks = int(mask_positions.sum().item())
            reveal_count = max(current_masks - remaining_target, 0)
            reveal_count = min(reveal_count, current_masks)

            if reveal_count > 0:
                masked_indices = mask_positions[0].nonzero(as_tuple=False).reshape(-1)
                masked_conf = confidence[0, masked_indices]
                chosen = masked_indices[torch.topk(masked_conf, k=reveal_count).indices]
                xt[0, chosen] = pred[0, chosen]

        if step in {num_steps - 1, (3 * num_steps) // 4, num_steps // 2, num_steps // 4, 0}:
            trajectory.append((step, xt[0].clone()))

    return xt, trajectory


def main():
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    ckpt_args = checkpoint["args"]

    model = DiscreteDenoiser(
        vocab_size=ckpt_args["vocab_size"],
        seq_len=ckpt_args["seq_len"],
        hidden_dim=ckpt_args["hidden_dim"],
        num_layers=ckpt_args["num_layers"],
        num_heads=ckpt_args["num_heads"],
        num_steps=ckpt_args["num_steps"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])

    sample, trajectory = sample_discrete(
        model=model,
        seq_len=ckpt_args["seq_len"],
        num_steps=ckpt_args["num_steps"],
        mask_token_id=ckpt_args["mask_token_id"],
        device=device,
    )

    for step, seq in trajectory:
        print(f"step={step}: {format_sequence(seq)}")
    print(f"final: {format_sequence(sample[0])}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table = ax.table(
        cellText=[[f"step={step}", format_sequence(seq)] for step, seq in trajectory],
        colLabels=["reverse step", "sequence"],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.5)
    ax.set_title("Discrete reverse denoising process")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reverse process figure to: {save_path}")


if __name__ == "__main__":
    main()
