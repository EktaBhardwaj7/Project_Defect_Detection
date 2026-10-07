# member2_supervised/engine.py
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             average_precision_score)
from config import DEVICE


def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion=None, use_tta=False, return_preds=False):
    model.eval()
    total_loss, n = 0.0, 0
    all_labels, all_preds, all_probs, all_paths = [], [], [], []

    for batch in tqdm(loader, desc="eval", leave=False):
        if len(batch) == 3:
            imgs, labels, paths = batch
            all_paths.extend(list(paths))
        else:
            imgs, labels = batch
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        if use_tta:
            logits = 0
            for dims in (None, [3], [2], [2, 3]):
                x = imgs if dims is None else torch.flip(imgs, dims=dims)
                logits = logits + model(x)
            logits = logits / 4.0
        else:
            logits = model(imgs)

        probs = torch.softmax(logits, dim=1)[:, 1]

        if criterion is not None:
            total_loss += criterion(logits, labels).item() * imgs.size(0)
        n += imgs.size(0)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(logits.argmax(1).cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

    y  = np.array(all_labels)
    p  = np.array(all_preds)
    pr = np.array(all_probs)

    metrics = {
        "loss": float(total_loss / n) if criterion is not None else None,
        "accuracy":  float(accuracy_score(y, p)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall":    float(recall_score(y, p, zero_division=0)),
        "f1":        float(f1_score(y, p, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, p).tolist(),
        "n_samples": int(n),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y, pr))
    except ValueError:
        metrics["auroc"] = 0.0
    try:
        metrics["ap"] = float(average_precision_score(y, pr))
    except ValueError:
        metrics["ap"] = 0.0

    if return_preds:
        metrics["_preds"] = {"y_true": y.tolist(),
                             "y_pred": p.tolist(),
                             "y_prob": pr.tolist(),
                             "paths": all_paths}
    return metrics