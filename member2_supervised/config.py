"""
config.py — paths + hyper-parameters for member2_supervised.
"""
from pathlib import Path

PKG_DIR   = Path(__file__).resolve().parent
WORKSPACE = PKG_DIR.parent

METADATA_DIR = WORKSPACE / 'metadata'
MVTEC_META   = METADATA_DIR / 'mvtec_metadata.csv'
VISA_META    = METADATA_DIR / 'visa_metadata.csv'

def _looks_like_root(p: Path) -> bool:
    if not p.is_dir():
        return False
    return ((p / 'MVTec_AD').is_dir() or
            (p / 'VisA').is_dir() or
            (p / 'mvtec_anomaly_detection').is_dir() or
            (p / 'VisA_20220922').is_dir())

def _find_root() -> Path:
    candidates = [
        Path('/content/drive/MyDrive/IndustrialDefectProject_Datasets'),
        Path('/content/drive/MyDrive/IndustrialDefectProject/datasets'),
        Path('/content/drive/MyDrive/IndustrialDefectProject'),
        Path('/content/drive/MyDrive/datasets'),
        Path('/content/datasets'),
        WORKSPACE / 'datasets',
    ]
    for c in candidates:
        if _looks_like_root(c):
            return c
    drive = Path('/content/drive/MyDrive')
    if drive.is_dir():
        for p in drive.rglob('MVTec_AD'):
            if p.is_dir():
                return p.parent
    return candidates[0]

DATASETS_DIR = _find_root()

def _pick(*cands: Path) -> Path:
    for c in cands:
        if c.is_dir():
            return c
    return cands[0]

MVTEC_ROOT = _pick(
    DATASETS_DIR / 'MVTec_AD' / 'mvtec_anomaly_detection',
    DATASETS_DIR / 'MVTec_AD',
    DATASETS_DIR / 'mvtec_anomaly_detection',
)
VISA_ROOT = _pick(
    DATASETS_DIR / 'VisA' / 'VisA_20220922',
    DATASETS_DIR / 'VisA',
    DATASETS_DIR / 'VisA_20220922',
)

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

SEED        = 42
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
ARCHS          = ['convnext_tiny']

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
