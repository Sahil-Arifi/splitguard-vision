"""Compact CIFAR-10 classifier used for controlled integrity experiments."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn


class TinyCifarClassifier(nn.Module):
    """A small CNN whose purpose is controlled comparison, not SOTA accuracy."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be an integer")
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images: Tensor) -> Tensor:
        """Return class logits for a normalized NCHW image batch."""

        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
            raise ValueError("images must have NCHW shape (batch, 3, 32, 32)")
        if not torch.is_floating_point(images):
            raise TypeError("images must be floating-point normalized tensors")
        features = self.features(images)
        return cast(Tensor, self.classifier(torch.flatten(features, 1)))


__all__ = ["TinyCifarClassifier"]
