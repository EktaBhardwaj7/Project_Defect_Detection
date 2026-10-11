"""cross_dataset.py — train on one dataset, test on the other (domain shift).

Usage:
    python cross_dataset.py --run mvtec_resnet18 --target visa
    python cross_dataset.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import ARCHS, NUM_CLASSES, OUT_DIR, print_config
from dataset import eval_transform, make_loader
from engine import evaluate, get_device, save_json
from losses import get_loss
from model import build_model
from train import load_split

DATASETS = ['mvtec', 'visa']


def _parse_run_name(run_name: str):
    parts = run_name.split('_', 1)
    if len(parts) != 2:
        raise ValueError(f'bad run name {run_name!r}')
    return parts[0], parts[1]


def cross_eval(run_name: str, target_dataset: str, split: str = 'test',
               device=None, verbose: bool = True) -> dict:
    """Load a checkpoint trained on the source dataset, evaluate on target."""
    run_dir = Path(OUT_DIR) / run_name
    ckpt_path = run_dir / f'{run_name}_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    meta = ckpt.get('meta', {}) or {}
    source = meta.get('dataset') or _parse_run_name(run_name)[0]
    arch = meta.get('arch') or _parse_run_name(run_name)[1]

    if target_dataset == source:
        raise ValueError('source and target datasets are the same')

    device = device or get_device()
    model = build_model(arch, NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    df = load_split(target_dataset, split)
    loader = make_loader(df, eval_transform(), shuffle=False, device=device)
    criterion = get_loss('ce')

    res = evaluate(model, loader, criterion, device)
    out = {'run': run_name, 'source': source, 'target': target_dataset,
           'arch': arch, 'split': split, 'n': int(len(df)), **res}

    tag = f'cross_{source}_to_{target_dataset}_{arch}'
    save_json(out, Path(OUT_DIR) / f'{tag}.json')

    if verbose:
        print(f'\n[cross] {source} -> {target_dataset} ({arch})  n={out["n"]}  '
              f'acc={out["accuracy"]:.4f}  f1={out["f1"]:.4f}  '
              f'auc={out["auc"]:.4f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=None, help='e.g. mvtec_resnet18')
    ap.add_argument('--target', default=None, choices=DATASETS)
    ap.add_argument('--split', default='test',
                    choices=['train', 'val', 'test'])
    ap.add_argument('--all', action='store_true',
                    help='run every source/target/arch combination')
    args = ap.parse_args()

    print_config()

    if args.all:
        rows = []
        for src in DATASETS:
            tgt = 'visa' if src == 'mvtec' else 'mvtec'
            for arch in ARCHS:
                run_name = f'{src}_{arch}'
                try:
                    r = cross_eval(run_name, tgt, args.split)
                except FileNotFoundError as e:
                    print(f'  [skip] {e}')
                    continue
                rows.append({'source': src, 'target': tgt, 'arch': arch,
                             'n': r['n'], 'accuracy': r['accuracy'],
                             'f1': r['f1'], 'auc': r['auc']})
        if rows:
            import pandas as pd
            out_csv = Path(OUT_DIR) / 'cross_dataset.csv'
            pd.DataFrame(rows).to_csv(out_csv, index=False)
            print(f'\nwrote {out_csv}')
        return

    if not (args.run and args.target):
        ap.error('need --run and --target, or --all')

    res = cross_eval(args.run, args.target, args.split)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()