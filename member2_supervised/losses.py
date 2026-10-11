"""losses.py — class-balanced CE and focal loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FOCAL_GAMMA, NUM_CLASSES


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None,
                 reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if weight is not None:
            self.register_buffer('weight', weight.float())
        else:
            self.weight = None

    def forward(self, logits, targets):
        logp  = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt    = logpt.exp()
        loss  = -((1.0 - pt) ** self.gamma) * logpt

        if self.weight is not None:
            loss = loss * self.weight.to(logits.device).gather(0, targets)

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def class_weights_from_df(df, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    counts = df['label'].value_counts().to_dict()
    n = max(len(df), 1)
    w = []
    for c in range(num_classes):
        nc = max(int(counts.get(c, 0)), 1)
        w.append(n / (num_classes * nc))
    return torch.tensor(w, dtype=torch.float32)


def get_loss(name: str = 'ce_balanced', class_weights: torch.Tensor = None):
    name = (name or '').lower()
    if name in ('ce', 'cross_entropy', 'crossentropy'):
        return nn.CrossEntropyLoss()
    if name in ('ce_balanced', 'ce_weighted', 'weighted_ce'):
        return nn.CrossEntropyLoss(weight=class_weights)
    if name in ('focal', 'focal_loss'):
        return FocalLoss(gamma=FOCAL_GAMMA, weight=class_weights)
    raise ValueError(f'Unknown loss {name!r}')