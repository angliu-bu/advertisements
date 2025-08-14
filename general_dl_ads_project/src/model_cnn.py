"""CNN-based classifier for advertisement images.

Example:
    >>> from .model_cnn import AdImageCNN
    >>> model = AdImageCNN(model_name='resnet18', num_classes=2)
"""

from __future__ import annotations

from typing import Dict

import torch.nn as nn
from torchvision import models


_BACKBONES: Dict[str, callable] = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "efficientnet_b0": models.efficientnet_b0,
}


class AdImageCNN(nn.Module):
    """Wrapper around torchvision backbones for ad image classification."""

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 2,
        pretrained: bool = True,
        fine_tune: bool = True,
    ) -> None:
        super().__init__()
        if model_name not in _BACKBONES:
            raise ValueError(f"Unsupported model name: {model_name}")

        weights = "DEFAULT" if pretrained else None
        backbone_fn = _BACKBONES[model_name]
        self.model = backbone_fn(weights=weights)

        if model_name.startswith("resnet"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
            final_params = self.model.fc.parameters()
        elif model_name.startswith("efficientnet"):
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, num_classes)
            final_params = self.model.classifier[1].parameters()
        else:  # pragma: no cover - handled by check above
            final_params = []

        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in final_params:
                param.requires_grad = True

    def forward(self, x):  # pragma: no cover - simple wrapper
        return self.model(x)
