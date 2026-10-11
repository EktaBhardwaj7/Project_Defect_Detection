
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

try:
    from .dataset import load_metadata, DefectDataset
    from .model import build_model
except ImportError:
    from dataset import load_metadata, DefectDataset
    from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
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
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size

    return total_loss / max(total, 1), correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--root", action="append", default=[])
    parser.add_argument("--output-dir", default="member2_output")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_metadata(args.train_csv, args.root, strict=True)
    val_df = load_metadata(args.val_csv, args.root, strict=True)

    counts = train_df["label"].value_counts().reindex([0, 1], fill_value=0)
    if (counts == 0).any():
        raise ValueError("Training CSV must contain both classes.")

    train_ds = DefectDataset(train_df, args.image_size, augment=True)
    val_ds = DefectDataset(val_df, args.image_size, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == "cuda"
    )

    class_weights = len(train_df) / (2.0 * counts.to_numpy())
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )

    model = build_model(
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )

    best_val_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

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
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "image_size": args.image_size,
                "class_names": ["normal", "defective"],
            }, output_dir / "best_model.pt")
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            print("Early stopping.")
            break

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.csv", index=False
    )
    with open(output_dir / "training_config.json", "w") as file:
        json.dump(vars(args), file, indent=2)

    print("Best checkpoint:", output_dir / "best_model.pt")
    print("History:", output_dir / "training_history.csv")


if __name__ == "__main__":
    main()
