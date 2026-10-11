"""ConvNeXt-Tiny model and checkpoint utilities."""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(
    pretrained=True,
    dropout=0.30,
    freeze_backbone=False,
):
    """Build a binary ConvNeXt-Tiny classifier."""

    weights = (
        models.ConvNeXt_Tiny_Weights.DEFAULT
        if pretrained
        else None
    )

    model = models.convnext_tiny(weights=weights)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[2].in_features

    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 2),
    )

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    return model


def load_checkpoint(checkpoint_path, device="cpu"):
    """Load a checkpoint created by this project."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    config = checkpoint.get("model_config", {})

    model = build_model(
        pretrained=False,
        dropout=float(config.get("dropout", 0.30)),
        freeze_backbone=False,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint