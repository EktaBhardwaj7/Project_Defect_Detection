"""evaluate.py — evaluate a saved checkpoint on a split.

Usage:
    python evaluate.py --run mvtec_convnext_tiny --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from config import NUM_CLASSES, OUT_DIR, print_config
from dataset import eval_transform, make_loader
from engine import evaluate, get_device, save_json
from losses import get_loss
from model import build_model
from train import load_split


def _split_run_name(run_name: str):
    parts = run_name.split('_', 1)
    if len(parts) != 2:
        raise ValueError(f'run name must look like "<dataset>_<arch>", '
                         f'got {run_name!r}')
    return parts[0], parts[1]


def evaluate_run(run_name: str, split: str = 'test', device=None,
                 verbose: bool = True) -> dict:
    run_dir   = Path(OUT_DIR) / run_name
    ckpt_path = run_dir / f'{run_name}_best.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'checkpoint not found: {ckpt_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    meta = ckpt.get('meta', {}) or {}
    dataset = meta.get('dataset') or _split_run_name(run_name)[0]
    arch    = meta.get('arch')    or _split_run_name(run_name)[1]

    device = device or get_device()
    model  = build_model(arch, NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    df     = load_split(dataset, split)
    loader = make_loader(df, eval_transform(), shuffle=False, device=device)
    crit   = get_loss('ce')

    res = evaluate(model, loader, crit, device)
    out = {'run': run_name, 'dataset': dataset, 'arch': arch, 'split': split,
           'n': int(len(df)), 'ckpt_epoch': ckpt.get('epoch'),
           'ckpt_val_f1': ckpt.get('val_f1'), **res}

    save_json(out, run_dir / f'metrics_{split}.json')

    cm = pd.DataFrame(out['confusion_matrix'],
                      index=['true_0', 'true_1'],
                      columns=['pred_0', 'pred_1'])
    cm.to_csv(run_dir / f'confusion_{split}.csv')

    if verbose:
        print(f'\n[{run_name} / {split}]  n={out["n"]}  '
              f'loss={out["loss"]:.4f}  acc={out["accuracy"]:.4f}  '
              f'f1={out["f1"]:.4f}  prec={out["precision"]:.4f}  '
              f'rec={out["recall"]:.4f}  auc={out["auc"]:.4f}')
        print(cm.to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--split', default='test',
                    choices=['train', 'val', 'test'])
    args = ap.parse_args()

    print_config()
    res = evaluate_run(args.run, args.split)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()