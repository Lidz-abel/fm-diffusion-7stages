from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3


@dataclass
class ImageMetricStats:
    mean: torch.Tensor
    covariance: torch.Tensor
    logits: torch.Tensor | None = None


class InceptionFeatureExtractor(torch.nn.Module):
    """
    InceptionV3 feature/logit extractor for CIFAR-10 sample evaluation.

    Inputs are expected in [-1, 1] or [0, 1]. Set input_range accordingly.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = Inception_V3_Weights.DEFAULT if pretrained else None
        self.model = inception_v3(weights=weights, transform_input=False, aux_logits=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor, input_range: str = "minus_one_one") -> tuple[torch.Tensor, torch.Tensor]:
        if input_range == "minus_one_one":
            images = (images + 1.0) * 0.5
        elif input_range != "zero_one":
            raise ValueError("input_range must be 'minus_one_one' or 'zero_one'.")
        images = images.clamp(0.0, 1.0)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)

        x = self.model.Conv2d_1a_3x3(images)
        x = self.model.Conv2d_2a_3x3(x)
        x = self.model.Conv2d_2b_3x3(x)
        x = self.model.maxpool1(x)
        x = self.model.Conv2d_3b_1x1(x)
        x = self.model.Conv2d_4a_3x3(x)
        x = self.model.maxpool2(x)
        x = self.model.Mixed_5b(x)
        x = self.model.Mixed_5c(x)
        x = self.model.Mixed_5d(x)
        x = self.model.Mixed_6a(x)
        x = self.model.Mixed_6b(x)
        x = self.model.Mixed_6c(x)
        x = self.model.Mixed_6d(x)
        x = self.model.Mixed_6e(x)
        x = self.model.Mixed_7a(x)
        x = self.model.Mixed_7b(x)
        x = self.model.Mixed_7c(x)
        features = self.model.avgpool(x)
        features = torch.flatten(features, 1)
        logits = self.model.fc(self.model.dropout(features))
        return features, logits


def compute_stats(features: torch.Tensor, logits: torch.Tensor | None = None) -> ImageMetricStats:
    features = features.float()
    mean = features.mean(dim=0)
    centered = features - mean
    denom = max(features.shape[0] - 1, 1)
    covariance = centered.T @ centered / denom
    return ImageMetricStats(mean=mean, covariance=covariance, logits=logits)


def frechet_distance(real: ImageMetricStats, fake: ImageMetricStats, eps: float = 1e-6) -> float:
    """
    Compute FID using a symmetric PSD formulation:
        Tr(sigma_r + sigma_f - 2 * sqrt(sqrt(sigma_r) sigma_f sqrt(sigma_r))).
    """
    device = real.mean.device
    dtype = torch.float64
    mu1 = real.mean.to(device=device, dtype=dtype)
    mu2 = fake.mean.to(device=device, dtype=dtype)
    sigma1 = real.covariance.to(device=device, dtype=dtype)
    sigma2 = fake.covariance.to(device=device, dtype=dtype)
    eye = torch.eye(sigma1.shape[0], device=device, dtype=dtype)
    sigma1 = sigma1 + eps * eye
    sigma2 = sigma2 + eps * eye

    eigvals1, eigvecs1 = torch.linalg.eigh(sigma1)
    eigvals1 = eigvals1.clamp_min(0.0)
    sqrt_sigma1 = (eigvecs1 * eigvals1.sqrt().unsqueeze(0)) @ eigvecs1.T

    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    middle = 0.5 * (middle + middle.T)
    eigvals_middle = torch.linalg.eigvalsh(middle).clamp_min(0.0)
    trace_sqrt = eigvals_middle.sqrt().sum()

    diff = mu1 - mu2
    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2.0 * trace_sqrt
    return float(fid.clamp_min(0.0).item())


def inception_score(logits: torch.Tensor, splits: int = 10) -> tuple[float, float]:
    logits = logits.float()
    probs = torch.softmax(logits, dim=1)
    n = probs.shape[0]
    splits = max(1, min(splits, n))
    scores = []
    for part in torch.chunk(probs, splits, dim=0):
        py = part.mean(dim=0, keepdim=True)
        kl = part * (torch.log(part.clamp_min(1e-8)) - torch.log(py.clamp_min(1e-8)))
        scores.append(torch.exp(kl.sum(dim=1).mean()))
    stacked = torch.stack(scores)
    return float(stacked.mean().item()), float(stacked.std(unbiased=False).item())
