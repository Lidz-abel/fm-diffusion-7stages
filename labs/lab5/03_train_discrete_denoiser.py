from pathlib import Path
import argparse
import sys

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.discrete_dataset import ToySequenceDataset
from src.discrete_denoiser import DiscreteDenoiser
from src.discrete_diffusion import MaskCorruption, MaskDiffusionSchedule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--mask_token_id", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default=str(ROOT / "checkpoints" / "discrete_denoiser_toy.pt"))
    parser.add_argument("--masked_only", action="store_true")
    return parser.parse_args()


def compute_loss(logits: torch.Tensor, x0: torch.Tensor, mask: torch.Tensor, masked_only: bool) -> torch.Tensor:
    if masked_only and mask.any():
        return F.cross_entropy(logits[mask], x0[mask])
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x0.reshape(-1))


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = ToySequenceDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    schedule = MaskDiffusionSchedule(num_steps=args.num_steps).to(device)
    corruption = MaskCorruption(schedule=schedule, mask_token_id=args.mask_token_id)
    model = DiscreteDenoiser(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_steps=args.num_steps,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    loss_history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for x0 in pbar:
            x0 = x0.to(device)
            t = torch.randint(0, args.num_steps, (x0.shape[0],), device=device)
            xt, mask = corruption.q_sample(x0, t)
            logits = model(xt, t)
            loss = compute_loss(logits, x0, mask, masked_only=args.masked_only)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"epoch={epoch} loss={avg_loss:.6f}")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "loss": loss_history,
        },
        save_path,
    )
    print(f"Saved discrete denoiser checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
