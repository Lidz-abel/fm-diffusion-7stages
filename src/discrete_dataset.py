import torch
from torch.utils.data import Dataset


class ToySequenceDataset(Dataset):
    """
    Toy sequence dataset for discrete diffusion.

    Token 0 is [MASK]. Valid data tokens start from 1.
    """

    def __init__(self, num_samples: int = 10_000, seq_len: int = 8, vocab_size: int = 10):
        if seq_len != 8:
            raise ValueError("This toy dataset currently expects seq_len=8.")
        if vocab_size < 9:
            raise ValueError("vocab_size must be at least 9 for tokens 0-8.")

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.patterns = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [8, 7, 6, 5, 4, 3, 2, 1],
                [1, 1, 2, 2, 3, 3, 4, 4],
                [1, 3, 5, 7, 2, 4, 6, 8],
            ],
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        pattern_id = torch.randint(0, len(self.patterns), (1,)).item()
        return self.patterns[pattern_id].clone()
