"""train.py — train one (dataset, arch) run.

Usage:
    python train.py --dataset mvtec --arch convnext_tiny
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (ARCHS, BATCH_SIZE, EPOCHS, LOSS_NAME, LR_BACKBONE,
                    LR_HEAD, NUM_CLASSES, OUT_DIR, SEED, SPLIT_FILES,
                    WEIGHT_DECAY, print_config)
from dataset import eval_transform, make_loader, train_transform
from engine import fit, get_device, save_json
from losses import class_weights_from_df, get_loss
from model import build_model, get_param_groups


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(dataset: str, split: str) -> pd.DataFrame:
    csv = Path(SPLIT_FILES[dataset][split])
    if not csv.exists():
        raise FileNotFoundError(
            f'{csv} not found — build splits first '
            f'(run_all.py --splits-only or the Colab script).')
    return pd.read_csv(csv)


def train_run(dataset: str = 'mvtec', arch: str = 'convnext_tiny',
              epochs: int = EPOCHS, loss_name: str = LOSS_NAME,
              batch_size: int = None, lr_backbone: float = LR_BACKBONE,
              lr_head: float = LR_HEAD, weight_decay: float = WEIGHT_DECAY,
              patience: int = 5, device=None, verbose: bool = True) -> dict:
    set_seed(SEED)
    device = device or get_device()
    bs = batch_size or BATCH_SIZE

    train_df = load_split(dataset, 'train')
    val_df   = load_split(dataset, 'val')
    if train_df['label'].nunique() < 2:
        raise RuntimeError(f'{dataset} train split has only one class')

    train_loader = make_loader(train_df, train_transform(), shuffle=True,
                               batch_size=bs, device=device)
    val_loader   = make_loader(val_df, eval_transform(), shuffle=False,
                               batch_size=bs, device=device)

    model = build_model(arch, NUM_CLASSES, pretrained=True).to(device)
    bb_p, hd_p = get_param_groups(model, arch)
    optimizer = Adam(
        [{'params': bb_p, 'lr': lr_backbone},
         {'params': hd_p, 'lr': lr_head}],
        weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    weights   = class_weights_from_df(train_df, NUM_CLASSES)
    criterion = get_loss(loss_name, weights.to(device))

    run_name = f'{dataset}_{arch}'
    run_dir  = Path(OUT_DIR) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    meta = {'dataset': dataset, 'arch': arch, 'loss': loss_name,
            'epochs': epochs, 'batch_size': bs,
            'lr_backbone': lr_backbone, 'lr_head': lr_head,
            'weight_decay': weight_decay,
            'num_train': int(len(train_df)), 'num_val': int(len(val_df)),
            'class_weights': weights.tolist()}

    if verbose:
        print(f'\n=== {run_name} ===')
        print(f'  train={len(train_df)}  val={len(val_df)}  device={device}')
        print(f'  class counts (train): '
              f'{train_df["label"].value_counts().to_dict()}')

    result = fit(model, train_loader, val_loader, criterion, optimizer, device,
                 epochs=epochs, scheduler=scheduler, ckpt_dir=run_dir,
                 run_name=run_name, patience=patience, meta=meta,
                 verbose=verbose)

    meta.update({'best_val_f1': result['best_f1'],
                 'best_epoch': result['best_epoch'],
                 'device': str(device)})
    save_json(result['history'], run_dir / 'history.json')
    save_json(meta, run_dir / 'metrics.json')

    if verbose:
        print(f'  best val F1 = {result["best_f1"]:.4f} '
              f'@ epoch {result["best_epoch"]}')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mvtec', 'visa'], default='mvtec')
    ap.add_argument('--arch', default=ARCHS[0])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--loss', default=LOSS_NAME)
    ap.add_argument('--batch-size', type=int, default=None)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    print_config()
    meta = train_run(args.dataset, args.arch, epochs=args.epochs,
                     loss_name=args.loss, batch_size=args.batch_size,
                     patience=args.patience, verbose=not args.quiet)
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()