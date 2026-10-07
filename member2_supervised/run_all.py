# member2_supervised/run_all.py
"""
Member 2 – Supervised Deep Learning
End-to-end runner for Colab.

Usage:
    %cd member2_supervised
    !python run_all.py

Before running:
    Upload mvtec_metadata.csv and visa_metadata.csv to ../metadata/
"""
import os, sys, shutil, json
from pathlib import Path

# Install deps
os.system("pip install -q kagglehub timm scikit-learn pandas tqdm")

import kagglehub
import config
from config import (set_seed, METADATA_DIR, OUTPUT_DIR, RESULTS_DIR,
                    MVTEC_ROOT as _M, VISA_ROOT as _V)
from train import train_supervised
from evaluate import evaluate_checkpoint
from cross_dataset import cross_dataset_experiment


def find_inner(base: Path, keyword: str) -> Path:
    if keyword.lower() in base.name.lower():
        return base
    for p in base.rglob("*"):
        if p.is_dir() and keyword.lower() in p.name.lower():
            return p
    raise FileNotFoundError(f"Cannot find '{keyword}' under {base}")


def main():
    set_seed(42)

    # ---- Download datasets ----
    print("Downloading MVTec ...")
    mvtec_path = Path(kagglehub.dataset_download("ipythonx/mvtec-ad"))
    print("Downloading VisA ...")
    visa_path  = Path(kagglehub.dataset_download("sorenasafari44/visa-dataset"))

    mvtec_root = find_inner(mvtec_path, "mvtec_anomaly_detection")
    visa_root  = find_inner(visa_path,  "VisA_20220922")
    print("MVTec root:", mvtec_root)
    print("VisA  root:", visa_root)

    # Patch config so downstream modules see correct roots
    config.MVTEC_ROOT = mvtec_root
    config.VISA_ROOT  = visa_root

    # ---- Metadata ----
    METADATA_DIR.mkdir(exist_ok=True, parents=True)
    mvtec_csv = METADATA_DIR / "mvtec_metadata.csv"
    visa_csv  = METADATA_DIR / "visa_metadata.csv"
    assert mvtec_csv.exists(), f"Missing {mvtec_csv}"
    assert visa_csv.exists(),  f"Missing {visa_csv}"

    summary = {}

    # ---- 1. In-domain training ----
    train_supervised("mvtec_supervised", mvtec_csv, mvtec_root)
    train_supervised("visa_supervised",  visa_csv,  visa_root)

    # ---- 2. In-domain test (with TTA) ----
    summary["mvtec_in_domain"] = evaluate_checkpoint(
        "mvtec_supervised", mvtec_csv, mvtec_root, split="test", use_tta=True)
    summary["visa_in_domain"] = evaluate_checkpoint(
        "visa_supervised",  visa_csv,  visa_root,  split="test", use_tta=True)

    # ---- 3. Cross-dataset ----
    summary["mvtec_to_visa"] = cross_dataset_experiment(
        "mvtec_to_visa",
        train_csv=mvtec_csv, train_root=mvtec_root,
        test_csv=visa_csv,   test_root=visa_root,
        tta=True)
    summary["visa_to_mvtec"] = cross_dataset_experiment(
        "visa_to_mvtec",
        train_csv=visa_csv,  train_root=visa_root,
        test_csv=mvtec_csv,  test_root=mvtec_root,
        tta=True)

    # ---- 4. Save master summary ----
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "="*70 + "\nFINAL SUMMARY\n" + "="*70)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"{k:20s} | Acc={v.get('accuracy',0):.4f} "
                  f"F1={v.get('f1',0):.4f} AUROC={v.get('auroc',0):.4f}")
    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")

    # ---- 5. Zip & download ----
    zip_path = shutil.make_archive("member2_supervised_output", "zip",
                                    root_dir=OUTPUT_DIR)
    print("Zipped:", zip_path)
    try:
        from google.colab import files
        files.download(zip_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()