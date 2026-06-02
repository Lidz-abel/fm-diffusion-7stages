from __future__ import annotations

import time
from dataclasses import dataclass

import torch


@dataclass
class TimingResult:
    seconds_total: float
    max_memory_mb: float

    @property
    def seconds_per_image(self) -> float:
        return self.seconds_total


def cuda_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def time_sampling(fn, device: torch.device) -> tuple[torch.Tensor, TimingResult]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    samples = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return samples, TimingResult(seconds_total=elapsed, max_memory_mb=cuda_memory_mb(device))
