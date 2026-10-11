
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, balanced_accuracy_score, matthews_corrcoef
)

try:
    from .dataset import load_metadata, DefectDataset
    from .model import load_checkpoint
except ImportError:
    from dataset import load_metadata, DefectDataset
    from model import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output-dir", default="evaluation_output")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_metadata(args.csv, args.root, strict=True)
    dataset = DefectDataset(df, args.image_size, augment=False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model, checkpoint = load_checkpoint(args.checkpoint, device)
    probabilities, labels = [], []

    with torch.no_grad():
        for images, batch_labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(probs.cpu().numpy())
            labels.extend(batch_labels.numpy())

    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= args.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "threshold": args.threshold,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_defective": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall_defective": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_defective": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {
            "true_normal": int(tn),
            "false_alarm": int(fp),
            "missed_defect": int(fn),
            "detected_defect": int(tp),
        },
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["average_precision"] = float(
            average_precision_score(y_true, y_prob)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    predictions = df.copy()
    predictions["true_label"] = y_true
    predictions["defect_probability"] = y_prob
    predictions["predicted_label"] = y_pred
    predictions.to_csv(output / "predictions.csv", index=False)

    with open(output / "metrics.json", "w") as file:
        json.dump(metrics, file, indent=2)

    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=["normal", "defective"], zero_division=0
    )
    (output / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(report)
    print("Evaluation outputs:", output)


if __name__ == "__main__":
    main()
