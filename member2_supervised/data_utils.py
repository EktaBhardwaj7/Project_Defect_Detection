"""
data_utils.py — reads Member 1's metadata, resolves image paths.

Member 1's metadata uses:
    datasets/mvtec_anomaly_detection/<cat>/...
    datasets/VisA_20220922/<cat>/...

Real Drive layout:
    <MVTEC_ROOT>/<cat>/...
    <VISA_ROOT>/<cat>/...
"""

from pathlib import Path
import pandas as pd

from config import MVTEC_ROOT, VISA_ROOT, DATASETS_DIR


def resolve_path(rel):
    if rel is None:
        return None
    if isinstance(rel, float) and pd.isna(rel):
        return None
    s = str(rel).strip()
    if s == '' or s.lower() in ('nan', 'none'):
        return None

    p = Path(s)
    if p.is_absolute() and p.exists():
        return p

    parts = list(p.parts)
    if parts and parts[0].lower() == 'datasets':
        parts = parts[1:]
    if not parts:
        return None

    top = parts[0]
    if top == 'mvtec_anomaly_detection':
        base = MVTEC_ROOT
    elif top == 'VisA_20220922':
        base = VISA_ROOT
    else:
        base = DATASETS_DIR

    cand = base / Path(*parts)
    return cand if cand.exists() else None


def load_metadata(csv_path) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if 'image_path' not in df.columns:
        raise KeyError(f"{csv_path.name}: no 'image_path' column "
                       f"(cols={list(df.columns)})")

    df['abs_path'] = df['image_path'].apply(resolve_path)
    n_missing = int(df['abs_path'].isna().sum())
    if n_missing:
        print(f'  [{csv_path.name}] dropping {n_missing} rows with missing images')
    df = df.dropna(subset=['abs_path']).reset_index(drop=True)

    if 'mask_path' in df.columns:
        df['mask_abs'] = df['mask_path'].apply(resolve_path)
    else:
        df['mask_abs'] = None
    return df


def pick_valid_category(df: pd.DataFrame, preferred: str) -> str:
    """Category that has both label 0 and label 1."""
    if 'category' not in df.columns:
        raise KeyError("metadata has no 'category' column")
    g = df.groupby('category')['label'].agg(['nunique', 'count'])
    valid = g[g['nunique'] >= 2].sort_values('count', ascending=False)
    if len(valid) == 0:
        raise RuntimeError('No category has both classes (0 and 1).')
    if preferred in valid.index:
        return preferred
    chosen = valid.index[0]
    print(f'  {preferred!r} not usable — falling back to {chosen!r}')
    print(f'  valid: {list(valid.index)}')
    return chosen