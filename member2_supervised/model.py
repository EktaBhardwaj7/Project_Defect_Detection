"""Train ConvNeXt-Tiny for normal/defective classification."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .dataset import load_metadata, DefectDataset
    from .model import build_model, load_checkpoint
    from .utils import set_seed
except ImportError:
    from dataset import load_metadata, DefectDataset
    from model import build_model, load_checkpoint
    from utils import set_seed


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional class weights."""

    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = float(gamma)

        # register_buffer(None) works in modern PyTorch, but we guard
        # explicitly so this also runs on older releases.
        if class_weights is None:
            self.class_weights = None
        else:
            self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        log_probs = torch.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()

        if self.class_weights is None:
            alpha = 1.0
        else:
            alpha = self.class_weights[targets]

        loss = -alpha * (1.0 - pt).pow(self.gamma) * log_pt
        return loss.mean()


def _make_grad_scaler(enabled: bool):
    """Create a GradScaler with fallbacks for different PyTorch versions."""
    try:
        # PyTorch >= 2.3 preferred API.
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        # Older PyTorch API.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device, enabled):
    """Return an autocast context for the given device."""
    try:
        return torch.autocast(device_type=device.type, enabled=enabled)
    except (TypeError, AttributeError):
        # Fallback for older PyTorch.
        return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_amp = device.type == "cuda"
    autocast_ctx = _autocast(device, enabled=use_amp)

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with autocast_ctx:
                logits = model(images)
                loss = criterion(logits, labels)

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("DataLoader returned no samples.")

    return (
        total_loss / total_samples,
        total_correct / total_samples,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    probabilities = []
    labels = []

    use_amp = device.type == "cuda"
    autocast_ctx = _autocast(device, enabled=use_amp)

    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)

        with autocast_ctx:
            logits = model(images)

        probs = torch.softmax(logits.float(), dim=1)[:, 1]

        probabilities.extend(probs.cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().tolist())

    return (
        np.asarray(labels, dtype=int),
        np.asarray(probabilities, dtype=float),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)

    parser.add_argument(
        "--root",
        action="append",
        default=[],
        help="Image root; repeat for multiple roots.",
    )

    parser.add_argument("--output-dir", default="training_output")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")

    args = parser.parse_args()

    if min(args.epochs, args.batch_size, args.patience) < 1:
        raise ValueError("epochs, batch-size and patience must be positive.")

    if args.warmup_epochs < 0 or args.focal_gamma < 0:
        raise ValueError("warmup-epochs and focal-gamma must be non-negative.")

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    output = os.path.abspath(args.output_dir)
    os.makedirs(output, exist_ok=True)

    train_df = load_metadata(args.train_csv, args.root, strict=True)
    val_df = load_metadata(args.val_csv, args.root, strict=True)

    # Guard against overlapping source train/validation images.
    train_paths = set(train_df["resolved_image_path"])
    val_paths = set(val_df["resolved_image_path"])

    overlap = train_paths & val_paths
    if overlap:
        raise ValueError(
            f"Train/validation leakage: {len(overlap)} shared images."
        )

    counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError("Training data must contain both classes.")

    train_ds = DefectDataset(train_df, args.image_size, augment=True)
    val_ds = DefectDataset(val_df, args.image_size, augment=False)

    workers = min(4, max(0, (os.cpu_count() or 2) - 1))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    # Inverse-frequency class weights, normalized to mean 1.
    weights = len(train_df) / (2.0 * counts.to_numpy(dtype=np.float32))
    weights = weights / weights.mean()

    criterion = FocalLoss(
        gamma=args.focal_gamma,
        class_weights=torch.tensor(
            weights, dtype=torch.float32, device=device
        ),
    )

    model_config = {
        "architecture": "convnext_tiny",
        "dropout": args.dropout,
        "num_classes": 2,
        "class_names": ["normal", "defective"],
    }

    model = build_model(
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))

    if warmup_epochs:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - warmup_epochs),
            eta_min=args.lr * 0.01,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )

    scaler = _make_grad_scaler(enabled=device.type == "cuda")

    best_val_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device,
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

        history.append(record)
        print(json.dumps(record))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": model_config,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "image_size": args.image_size,
                    "class_names": model_config["class_names"],
                    "training_args": vars(args),
                },
                os.path.join(output, "best_model.pt"),
            )
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(
        os.path.join(output, "training_history.csv"), index=False,
    )

    with open(
        os.path.join(output, "training_config.json"), "w", encoding="utf-8"
    ) as file:
        json.dump(vars(args), file, indent=2)

    # Generate validation predictions from the BEST checkpoint.
    best_model, _ = load_checkpoint(
        os.path.join(output, "best_model.pt"), device,
    )

    y_val, p_val = predict(best_model, val_loader, device)

    val_predictions = val_df.copy()
    val_predictions["true_label"] = y_val
    val_predictions["defect_probability"] = p_val

    val_predictions.to_csv(
        os.path.join(output, "val_predictions.csv"), index=False,
    )

    print("Best checkpoint:", os.path.join(output, "best_model.pt"))
    print("Validation predictions:", os.path.join(output, "val_predictions.csv"))


if __name__ == "__main__":
    main()