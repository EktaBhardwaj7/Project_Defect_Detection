"""
data_utils.py — your own copy of CSV / path helpers.
Reads metadata CSVs (read-only), resolves image paths against DATASETS_DIR.
"""

from pathlib import Path
import pandas as pd

from config import DATASETS_DIR


def resolve_path(rel):
    """
    Turn a relative path from the metadata CSV into an absolute Path.
    Tries DATASETS_DIR first, then DATASETS_DIR.parent.
    """
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

    parts = p.parts
    if parts and parts[0] == 'datasets':
        p = Path(*parts[1:])

    for base in (DATASETS_DIR, DATASETS_DIR.parent):
        cand = base / p
        if cand.exists():
            return cand
    return None


def load_metadata(csv_path) -> pd.DataFrame:
    """Read metadata CSV, resolve paths, drop rows whose image is missing."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        raise KeyError(f"{csv_path.name} has no 'image_path' column "
                       f"(columns: {list(df.columns)})")

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
    """Return a category that has both label=0 and label=1."""
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