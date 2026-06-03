from __future__ import annotations

import copy

import torch


class EMA:
    """
    Exponential moving average for model weights.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1).")
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        shadow_state = self.shadow.state_dict()
        for key, value in shadow_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                value.copy_(model_value)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "model": self.shadow.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.shadow.load_state_dict(state["model"])
