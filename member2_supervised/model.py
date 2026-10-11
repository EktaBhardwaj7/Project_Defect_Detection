"""model.py — torchvision backbones with a fresh binary head."""

import torch.nn as nn
from torchvision import models

from config import NUM_CLASSES

# Which module is the classification head for each arch
HEAD_MODULE = {
    'resnet18': 'fc',
    'resnet34': 'fc',
    'resnet50': 'fc',
    'mobilenet_v3_small': 'classifier',
    'mobilenet_v3_large': 'classifier',
}

try:  # torchvision >= 0.13
    models.resnet18(weights=None)
    _HAS_WEIGHTS_ARG = True
except TypeError:  # pragma: no cover
    _HAS_WEIGHTS_ARG = False


def _w(pretrained: bool):
    if _HAS_WEIGHTS_ARG:
        return {'weights': 'DEFAULT' if pretrained else None}
    return {'pretrained': bool(pretrained)}


def build_model(arch: str = 'resnet18', num_classes: int = NUM_CLASSES,
                pretrained: bool = True) -> nn.Module:
    arch = arch.lower()

    if arch in ('resnet18', 'resnet34', 'resnet50'):
        m = getattr(models, arch)(**_w(pretrained))
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if arch in ('mobilenet_v3_small', 'mobilenet_v3_large'):
        m = getattr(models, arch)(**_w(pretrained))
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f'Unsupported arch {arch!r}. '
                     f'Choose from {sorted(HEAD_MODULE)}')


def get_param_groups(model: nn.Module, arch: str):
    """Split trainable params into (backbone, head) for differential LRs."""
    arch = arch.lower()
    if arch not in HEAD_MODULE:
        raise ValueError(f'Unsupported arch {arch!r}')
    head_prefix = HEAD_MODULE[arch]

    backbone, head = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith(head_prefix) else backbone).append(p)
    return backbone, head