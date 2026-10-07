# member2_supervised/model.py
import torch
import torch.nn as nn
import timm


class DefectClassifier(nn.Module):
    """
    Wraps any timm backbone (convnext_tiny, resnet50, efficientnet_b3 ...).
    """
    def __init__(self, model_name="convnext_tiny", num_classes=2,
                 pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))