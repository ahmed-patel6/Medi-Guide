"""PyTorch CNN architecture for binary bone-fracture classification."""

from __future__ import annotations

import torch
from torch import nn

from src.bone_fracture_detection.config import XRAY_IMAGE_SIZE


class BoneFractureCNN(nn.Module):
    """Five-stage convolutional network returning class logits."""

    def __init__(self) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1

        for filters in (32, 64, 128, 256, 512):
            for _ in range(2):
                blocks.extend(
                    [
                        nn.Conv2d(
                            in_channels,
                            filters,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(filters),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = filters
            blocks.append(nn.MaxPool2d(kernel_size=2))

        self.features = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 2),
        )
        self.input_size = XRAY_IMAGE_SIZE

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return two class logits; apply softmax only when reporting scores."""
        features = self.features(images)
        return self.classifier(self.global_pool(features))


def build_fracture_cnn() -> BoneFractureCNN:
    """Build the fracture CNN."""
    return BoneFractureCNN()
