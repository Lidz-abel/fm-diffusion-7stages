import torch
import torch.nn as nn


class DiscreteDenoiser(nn.Module):
    """
    Minimal Transformer denoiser for masked token sequences.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        num_steps: int = 100,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_steps = num_steps

        self.token_emb = nn.Embedding(vocab_size, hidden_dim)
        self.pos_emb = nn.Embedding(seq_len, hidden_dim)
        self.time_emb = nn.Embedding(num_steps, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xt: [B, L]
            t: [B]

        Returns:
            logits: [B, L, vocab_size]
        """
        if xt.ndim != 2:
            raise ValueError(f"Expected xt with shape [B, L], got {xt.shape}.")
        if t.ndim != 1 or t.shape[0] != xt.shape[0]:
            raise ValueError(f"Expected t with shape [B], got {t.shape}.")

        batch_size, seq_len = xt.shape
        pos = torch.arange(seq_len, device=xt.device)[None, :].expand(batch_size, seq_len)

        h = self.token_emb(xt)
        h = h + self.pos_emb(pos)
        h = h + self.time_emb(t)[:, None, :]
        h = self.encoder(h)
        return self.output(h)
