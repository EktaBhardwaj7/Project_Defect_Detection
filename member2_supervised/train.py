from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from dataset import (
    SupervisedDefectDataset,
    build_eval_transform,
    build_train_transform,
)
from model import ResNet18DefectClassifier


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_class_weights(labels: pd.Series) -> torch.Tensor:
    counts = labels.value_counts().sort_index()

    weights = len(labels) / (
        2.0 * counts.astype(float)
    )

    return torch.tensor(
        weights.values,
        dtype=torch.float32,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            predictions = (
                probabilities >= 0.5
            ).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "auroc": roc_auc_score(
            y_true,
            y_prob,
        ),
    }

    return metrics


def train(args):
    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    train_dataset = SupervisedDefectDataset(
        args.train_csv,
        transform=build_train_transform(
            args.image_size
        ),
    )

    val_dataset = SupervisedDefectDataset(
        args.val_csv,
        transform=build_eval_transform(
            args.image_size
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    class_weights = compute_class_weights(
        train_dataset.metadata["label"]
    ).to(device)

    print("Class weights:", class_weights)

    model = ResNet18DefectClassifier(
        pretrained=True
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_f1 = -float("inf")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(1, args.epochs + 1):

        model.train()

        running_loss = 0.0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

        train_loss = (
            running_loss / len(train_dataset)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        scheduler.step(
            val_metrics["f1"]
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"val_auc={val_metrics['auroc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "class_weights": class_weights.cpu(),
            }

            torch.save(
                checkpoint,
                output_dir / "best_model.pt",
            )

            print(
                f"Saved best model "
                f"(F1={best_f1:.4f})"
            )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-csv",
        required=True,
    )

    parser.add_argument(
        "--val-csv",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/supervised",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)