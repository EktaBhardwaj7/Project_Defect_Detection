# member2_supervised/train.py
import copy, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from config import (IMG_SIZE, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, WARMUP_EPOCHS,
                    LABEL_SMOOTHING, NUM_WORKERS, DEVICE, CKPT_DIR, RESULTS_DIR,
                    MODEL_NAME, NUM_CLASSES, PRETRAINED, DROPOUT,
                    FOCAL_ALPHA, FOCAL_GAMMA, EARLY_STOP_PATIENCE)
from dataset import DefectDataset, build_transforms
from model import DefectClassifier
from losses import FocalLoss
from engine import train_one_epoch, evaluate


def make_loaders(csv_file, root_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                 val_frac=0.1):
    train_full = DefectDataset(csv_file, root_dir, split="train",
                               transform=build_transforms(img_size, True))
    val_full = DefectDataset(csv_file, root_dir, split="train",
                             transform=build_transforms(img_size, False))
    n = len(train_full)
    val_size = max(1, int(val_frac * n))
    train_size = n - val_size
    g = torch.Generator().manual_seed(42)
    idx_train, idx_val = torch.randperm(n, generator=g).split([train_size, val_size])
    train_subset = torch.utils.data.Subset(train_full, idx_train.tolist())
    val_subset   = torch.utils.data.Subset(val_full,   idx_val.tolist())

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    return train_loader, val_loader


def build_model():
    return DefectClassifier(MODEL_NAME, NUM_CLASSES, PRETRAINED, DROPOUT).to(DEVICE)


def build_optim_sched(model, epochs=EPOCHS):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - WARMUP_EPOCHS))
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[WARMUP_EPOCHS])
    return optimizer, scheduler


def train_supervised(tag, csv_file, root_dir, epochs=EPOCHS):
    print(f"\n{'='*70}\n Training: {tag}  |  backbone={MODEL_NAME}\n{'='*70}")

    train_loader, val_loader = make_loaders(csv_file, root_dir)

    model = build_model()
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                          label_smoothing=LABEL_SMOOTHING)
    optimizer, scheduler = build_optim_sched(model, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_f1, best_state, wait = -1.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        val_m = evaluate(model, val_loader, criterion, use_tta=False)
        scheduler.step()

        log = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
               "lr": optimizer.param_groups[0]["lr"], **val_m,
               "time_s": round(time.time() - t0, 1)}
        history.append(log)
        print(f"Epoch {epoch:02d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"| val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f} "
              f"val_auroc={val_m['auroc']:.4f} | {log['time_s']}s")

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, CKPT_DIR / f"{tag}.pth")
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                print(f"[{tag}] Early stopping at epoch {epoch}")
                break

    # Restore best for reporting
    if best_state is not None:
        model.load_state_dict(best_state)

    with open(RESULTS_DIR / f"history_{tag}.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[{tag}] Best val F1 = {best_f1:.4f}")
    return model, best_f1, history