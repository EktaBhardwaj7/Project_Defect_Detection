from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18DefectClassifier(nn.Module):
    """
    Image-level binary defect classifier.

    Output:
        logits of shape [batch_size, 2]

    Class mapping:
        0 -> normal
        1 -> defective
    """

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        self.backbone = resnet18(weights=weights)

        num_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_features, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)