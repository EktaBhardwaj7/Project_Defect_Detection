# member2_supervised/config.py
from pathlib import Path
import torch, random, numpy as np

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_DIR = PROJECT_ROOT.parent / "metadata"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_DIR  = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# These get overwritten at runtime by run_all.py after kagglehub download
MVTEC_ROOT = None
VISA_ROOT  = None

# ---------- Training ----------
IMG_SIZE        = 288
BATCH_SIZE      = 32
EPOCHS          = 20
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
WARMUP_EPOCHS   = 2
LABEL_SMOOTHING = 0.05
NUM_WORKERS     = 2
SEED            = 42
EARLY_STOP_PATIENCE = 5

# ---------- Model ----------
MODEL_NAME  = "convnext_tiny"   # change to "resnet50" if you prefer
NUM_CLASSES = 2
PRETRAINED  = True
DROPOUT     = 0.3

# ---------- Loss ----------
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ---------- Device ----------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False