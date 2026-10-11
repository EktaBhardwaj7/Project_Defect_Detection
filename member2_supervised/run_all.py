"""run_all.py — build splits from Member 1's metadata, then train + evaluate.

Usage:
    python run_all.py                # full pipeline
    python run_all.py --splits-only
    python run_all.py --skip-cross --epochs 5
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (ARCHS, EPOCHS, MVTEC_CATEGORY, MVTEC_META, OUT_DIR, SEED,
                    SPLIT_FILES, VISA_CATEGORY, VISA_META, print_config)
from data_utils import load_metadata, pick_valid_category

DATASETS = ['mvtec', 'visa']

SPLIT_COLS = ['dataset', 'category', 'defect_type', 'label',
              'image_path', 'mask_path', 'split', 'abs_path']


# ------------------------------------------------------------------
# 1) splits — read Member 1's metadata, write our own CSVs
# ------------------------------------------------------------------
def _build_one(meta_csv, dataset_key, preferred_cat,
               val_frac=0.15, test_frac=0.15):
    """
    Read metadata (read-only), pick category, produce supervised splits.

    Auto strategy:
      - If Member 1's native 'train' split contains BOTH classes → use it,
        carving val from it. Keep Member 1's native 'test' as test.
      - Else (MVTec: native train is only class 0) → pool everything and
        do a stratified (1 - val - test, val, test) split.
    """
    df  = load_metadata(meta_csv)
    cat = pick_valid_category(df, preferred_cat)
    sub = df[df['category'] == cat].copy().reset_index(drop=True)

    print(f'[{dataset_key}] category={cat}  total={len(sub)}  '
          f'labels={sub["label"].value_counts().to_dict()}')

    tr_pool = sub[sub['split'] == 'train']
    te_pool = sub[sub['split'] == 'test']

    use_native = (
        len(tr_pool) >= 50
        and tr_pool['label'].nunique() >= 2
        and len(te_pool) >= 20
        and te_pool['label'].nunique() >= 2
    )

    if use_native:
        print('  → using native train/test split (val carved from train)')
        tr, va = train_test_split(tr_pool, test_size=val_frac,
                                  stratify=tr_pool['label'],
                                  random_state=SEED)
        te = te_pool
    else:
        print('  → native train has only one class → pooling + re-split')
        tr, tmp = train_test_split(sub, test_size=val_frac + test_frac,
                                   stratify=sub['label'],
                                   random_state=SEED)
        va, te = train_test_split(tmp,
                                  test_size=test_frac / (val_frac + test_frac),
                                  stratify=tmp['label'],
                                  random_state=SEED)

    paths = SPLIT_FILES[dataset_key]
    for name, d in [('train', tr), ('val', va), ('test', te)]:
        cols = [c for c in SPLIT_COLS if c in d.columns]
        d[cols].reset_index(drop=True).to_csv(paths[name], index=False)
        print(f'  {paths[name].name:35s} n={len(d):4d} '
              f'{d["label"].value_counts().to_dict()}')
    return cat


def make_splits():
    print('\n--- building splits from Member 1 metadata ---')
    m = _build_one(MVTEC_META, 'mvtec', MVTEC_CATEGORY)
    v = _build_one(VISA_META,  'visa',  VISA_CATEGORY)
    print(f'  MVTec category: {m}\n  VisA  category: {v}')


def splits_exist() -> bool:
    return all(Path(SPLIT_FILES[d][s]).exists()
               for d in DATASETS for s in ('train', 'val', 'test'))


# ------------------------------------------------------------------
# 2) main
# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-only', action='store_true')
    ap.add_argument('--skip-cross',  action='store_true')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()

    print_config()

    if args.splits_only or not splits_exist():
        make_splits()
    if args.splits_only:
        print('\nsplits built. done.')
        return

    from train import train_run
    from evaluate import evaluate_run
    from cross_dataset import cross_eval

    rows = []
    for ds in DATASETS:
        for arch in ARCHS:
            run_name = f'{ds}_{arch}'
            print(f'\n{"=" * 62}\n  TRAIN  {run_name}\n{"=" * 62}')
            try:
                meta = train_run(ds, arch, epochs=args.epochs)
                test = evaluate_run(run_name, split='test')
            except Exception:
                print(f'  [FAILED] {run_name}')
                traceback.print_exc()
                continue
            rows.append({'run': run_name, 'dataset': ds, 'arch': arch,
                         'best_val_f1': meta.get('best_val_f1'),
                         'test_loss': test['loss'],
                         'test_acc': test['accuracy'],
                         'test_f1': test['f1'],
                         'test_precision': test['precision'],
                         'test_recall': test['recall'],
                         'test_auc': test['auc']})
            pd.DataFrame(rows).to_csv(Path(OUT_DIR) / 'summary.csv', index=False)

    if not args.skip_cross:
        cross_rows = []
        for src in DATASETS:
            tgt = 'visa' if src == 'mvtec' else 'mvtec'
            for arch in ARCHS:
                run_name = f'{src}_{arch}'
                print(f'\n{"=" * 62}\n  CROSS  {src} -> {tgt}  ({arch})\n{"=" * 62}')
                try:
                    r = cross_eval(run_name, tgt, split='test')
                except Exception:
                    print(f'  [FAILED] {run_name} -> {tgt}')
                    traceback.print_exc()
                    continue
                cross_rows.append({'source': src, 'target': tgt, 'arch': arch,
                                   'n': r['n'], 'accuracy': r['accuracy'],
                                   'f1': r['f1'], 'auc': r['auc']})
                pd.DataFrame(cross_rows).to_csv(
                    Path(OUT_DIR) / 'cross_dataset.csv', index=False)

    print('\n' + '=' * 62)
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    print(f'\nartifacts in: {OUT_DIR}')


if __name__ == '__main__':
    main()