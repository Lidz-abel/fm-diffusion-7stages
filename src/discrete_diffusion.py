import torch


class MaskDiffusionSchedule:
    """
    Mask corruption schedule for discrete diffusion.

    beta_t is the per-step masking probability.
    alpha_bar_t is the probability that a token remains unmasked by step t.
    """

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 0.3,
        device: str | torch.device = "cpu",
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive.")
        if not 0.0 < beta_start < beta_end < 1.0:
            raise ValueError("expected 0 < beta_start < beta_end < 1.")

        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: str | torch.device) -> "MaskDiffusionSchedule":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self


class MaskCorruption:
    """
    Randomly replace tokens with [MASK].
    """

    def __init__(self, schedule: MaskDiffusionSchedule, mask_token_id: int = 0):
        self.schedule = schedule
        self.mask_token_id = mask_token_id

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x0: original tokens, shape [B, L]
            t: integer timesteps, shape [B]

        Returns:
            xt: corrupted tokens, shape [B, L]
            corruption_mask: True where token was replaced by [MASK]
        """
        if x0.ndim != 2:
            raise ValueError(f"Expected x0 with shape [B, L], got {x0.shape}.")
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError(f"Expected t with shape [B], got {t.shape}.")

        alpha_bar_t = self.schedule.alpha_bars.to(x0.device)[t]
        keep_prob = alpha_bar_t[:, None]
        keep_mask = torch.rand_like(x0.float()) < keep_prob

        xt = x0.clone()
        xt[~keep_mask] = self.mask_token_id
        return xt, ~keep_mask
