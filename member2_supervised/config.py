"""
config.py — paths + hyper-parameters for member2_supervised.

Rules:
- This file (and everything it points to) lives INSIDE member2_supervised/.
- metadata CSVs are READ-ONLY, in the sibling folder '../metadata/'.
- Image datasets live elsewhere on Drive — auto-detected, see DATASETS_DIR.
- splits/ and outputs/ are created INSIDE member2_supervised/.
"""

import sys
from pathlib import Path

# ------------------------------------------------------------------
# 1) Where am I?  (member2_supervised/)
# ------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent            # .../member2_supervised
WORKSPACE = PKG_DIR.parent                           # .../member2  (has sibling folders)

# ------------------------------------------------------------------
# 2) Metadata — read-only, sibling folder
# ------------------------------------------------------------------
METADATA_DIR = WORKSPACE / 'metadata'
MVTEC_META = METADATA_DIR / 'mvtec_metadata.csv'
VISA_META  = METADATA_DIR / 'visa_metadata.csv'

# ------------------------------------------------------------------
# 3) Datasets — search common Drive locations
# ------------------------------------------------------------------
def _looks_like_datasets_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    return ((p / 'mvtec_anomaly_detection').is_dir()
            or (p / 'VisA_20220922').is_dir()
            or (p / 'visa').is_dir())


def _find_datasets_root():
    candidates = [
        WORKSPACE / 'datasets',
        WORKSPACE.parent / 'datasets',
        Path('/content/drive/MyDrive/IndustrialDefectProject/datasets'),
        Path('/content/drive/MyDrive/datasets'),
        Path('/content/datasets'),
    ]
    for c in candidates:
        if _looks_like_datasets_root(c):
            return c

    drive_root = Path('/content/drive/MyDrive')
    if drive_root.exists():
        for p in drive_root.rglob('mvtec_anomaly_detection'):
            if p.is_dir():
                return p.parent

    return WORKSPACE / 'datasets'


DATASETS_DIR = _find_datasets_root()
MVTEC_ROOT   = DATASETS_DIR / 'mvtec_anomaly_detection'
VISA_ROOT    = DATASETS_DIR / 'VisA_20220922'

# ------------------------------------------------------------------
# 4) Outputs + splits — INSIDE your folder
# ------------------------------------------------------------------
SPLITS_DIR = PKG_DIR / 'splits'
OUT_DIR    = PKG_DIR / 'outputs'
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_FILES = {
    'mvtec': {
        'train': SPLITS_DIR / 'mvtec_supervised_train.csv',
        'val':   SPLITS_DIR / 'mvtec_supervised_val.csv',
        'test':  SPLITS_DIR / 'mvtec_supervised_test.csv',
    },
    'visa': {
        'train': SPLITS_DIR / 'visa_supervised_train.csv',
        'val':   SPLITS_DIR / 'visa_supervised_val.csv',
        'test':  SPLITS_DIR / 'visa_supervised_test.csv',
    },
}

# ------------------------------------------------------------------
# 5) Reproducibility + hyper-parameters
# ------------------------------------------------------------------
SEED = 42
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 2
NUM_CLASSES = 2
MEAN        = (0.485, 0.456, 0.406)
STD         = (0.229, 0.224, 0.225)

EPOCHS       = 15
LR_BACKBONE  = 1e-4
LR_HEAD      = 1e-3
WEIGHT_DECAY = 1e-4

LOSS_NAME   = 'ce_balanced'
FOCAL_GAMMA = 2.0

MVTEC_CATEGORY = 'bottle'
VISA_CATEGORY  = 'candle'
ARCHS          = ['resnet18', 'mobilenet_v3_small']


# ------------------------------------------------------------------
# 6) Debug helper
# ------------------------------------------------------------------
def print_config():
    print(f'PKG_DIR       : {PKG_DIR}')
    print(f'WORKSPACE     : {WORKSPACE}')
    print(f'METADATA_DIR  : {METADATA_DIR}   exists={METADATA_DIR.is_dir()}')
    print(f'  MVTEC_META  : {MVTEC_META}   exists={MVTEC_META.exists()}')
    print(f'  VISA_META   : {VISA_META}    exists={VISA_META.exists()}')
    print(f'DATASETS_DIR  : {DATASETS_DIR}   exists={DATASETS_DIR.is_dir()}')
    print(f'  MVTEC_ROOT  : {MVTEC_ROOT}   exists={MVTEC_ROOT.is_dir()}')
    print(f'  VISA_ROOT   : {VISA_ROOT}    exists={VISA_ROOT.is_dir()}')
    print(f'SPLITS_DIR    : {SPLITS_DIR}')
    print(f'OUT_DIR       : {OUT_DIR}')