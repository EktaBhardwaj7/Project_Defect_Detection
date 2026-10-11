"""engine.py — train/eval loops + metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

try:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    def make_scaler(enabled: bool):
        return _GradScaler('cuda', enabled=enabled)

    def autocast_ctx(enabled: bool):
        return _autocast('cuda', enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler as _GradScaler
    from torch.cuda.amp import autocast as _autocast

    def make_scaler(enabled: bool):
        return _GradScaler(enabled=enabled)

    def autocast_ctx(enabled: bool):
        return _autocast(enabled=enabled)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    m = {
        'accuracy':  float(accuracy_score(y_true, y_pred)),
        'f1':        float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred, zero_division=0)),
        'auc':       float('nan'),
    }
    if y_prob is not None and len(set(np.asarray(y_true).tolist())) > 1:
        try:
            m['auc'] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass
    m['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    return m


def _run_epoch(model, loader, criterion, device,
               optimizer=None, scaler=None, grad_clip=None):
    training = optimizer is not None
    model.train(training)

    total_loss, n = 0.0, 0
    labels, preds, probs = [], [], []

    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(training)
    try:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(scaler is not None):
                out = model(x)
                loss = criterion(out, y)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            n += bs

            p = torch.softmax(out.detach().float(), dim=1)
            probs.append(p.cpu().numpy())
            preds.append(p.argmax(dim=1).cpu().numpy())
            labels.append(y.detach().cpu().numpy())
    finally:
        torch.set_grad_enabled(prev)

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)[:, 1] if np.concatenate(probs).shape[1] > 1 \
             else np.concatenate(probs)[:, 0]

    return total_loss / max(n, 1), compute_metrics(y_true, y_pred, y_prob)


def evaluate(model, loader, criterion, device):
    loss, metrics = _run_epoch(model, loader, criterion, device)
    return {'loss': float(loss), **metrics}


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        epochs: int, scheduler=None, ckpt_dir=None, run_name: str = 'run',
        patience: int = 5, grad_clip: float = 5.0, use_amp: bool = True,
        meta: dict = None, verbose: bool = True):
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    scaler = make_scaler(use_amp and device.type == 'cuda')
    meta = meta or {}

    history = []
    best_f1, best_epoch, no_improve = -1.0, -1, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        tr_loss, tr_m = _run_epoch(model, train_loader, criterion, device,
                                   optimizer=optimizer, scaler=scaler,
                                   grad_clip=grad_clip)
        va_loss, va_m = _run_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        rec = {'epoch': epoch, 'lr': lr,
               'train_loss': tr_loss, 'val_loss': va_loss,
               'train_acc': tr_m['accuracy'], 'train_f1': tr_m['f1'],
               'val_acc': va_m['accuracy'], 'val_f1': va_m['f1'],
               'val_auc': va_m['auc'],
               'secs': round(time.time() - t0, 1)}
        history.append(rec)

        if verbose:
            print(f'  epoch {epoch:3d}/{epochs}  lr={lr:.2e}  '
                  f'train_loss={tr_loss:.4f} acc={tr_m["accuracy"]:.4f} f1={tr_m["f1"]:.4f}  |  '
                  f'val_loss={va_loss:.4f} acc={va_m["accuracy"]:.4f} '
                  f'f1={va_m["f1"]:.4f} auc={va_m["auc"]:.4f}  ({rec["secs"]}s)')

        score = va_m['f1']
        if score > best_f1:
            best_f1, best_epoch, no_improve = score, epoch, 0
            if ckpt_dir:
                torch.save({'model_state': model.state_dict(),
                            'epoch': epoch, 'val_f1': best_f1, 'meta': meta},
                           ckpt_dir / f'{run_name}_best.pt')
        else:
            no_improve += 1

        if ckpt_dir:
            torch.save({'model_state': model.state_dict(),
                        'epoch': epoch, 'val_f1': best_f1, 'meta': meta},
                       ckpt_dir / f'{run_name}_last.pt')

        if patience and no_improve >= patience:
            if verbose:
                print(f'  early stop at epoch {epoch}')
            break

    return {'history': history, 'best_f1': best_f1, 'best_epoch': best_epoch}


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))