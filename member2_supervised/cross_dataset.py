# member2_supervised/cross_dataset.py
import copy, json, time
import torch
from config import (EPOCHS, DEVICE, CKPT_DIR, RESULTS_DIR,
                    FOCAL_ALPHA, FOCAL_GAMMA, LABEL_SMOOTHING)
from dataset import DefectDataset, build_transforms
from torch.utils.data import DataLoader
from model import DefectClassifier
from losses import FocalLoss
from engine import train_one_epoch, evaluate
from train import build_optim_sched, make_loaders
from config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, MODEL_NAME, NUM_CLASSES, PRETRAINED, DROPOUT


def build_test_loader(csv_file, root_dir):
    ds = DefectDataset(csv_file, root_dir, split="test",
                       transform=build_transforms(IMG_SIZE, train=False),
                       return_path=True)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)


def cross_dataset_experiment(tag, train_csv, train_root, test_csv, test_root,
                             epochs=EPOCHS, tta=True):
    print(f"\n{'='*70}\n Cross-Dataset: {tag}\n{'='*70}")

    train_loader, val_loader = make_loaders(train_csv, train_root)
    test_loader = build_test_loader(test_csv, test_root)

    model = DefectClassifier(MODEL_NAME, NUM_CLASSES, PRETRAINED, DROPOUT).to(DEVICE)
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA,
                          label_smoothing=LABEL_SMOOTHING)
    optimizer, scheduler = build_optim_sched(model, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_f1, best_state, wait = -1.0, None, 0
    from config import EARLY_STOP_PATIENCE

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        val_m = evaluate(model, val_loader, criterion, use_tta=False)
        scheduler.step()
        print(f"Epoch {epoch:02d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} "
              f"| val_f1={val_m['f1']:.4f} val_auroc={val_m['auroc']:.4f} "
              f"| {time.time()-t0:.1f}s")
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

    if best_state is not None:
        model.load_state_dict(best_state)

    test_m = evaluate(model, test_loader, criterion=None, use_tta=tta, return_preds=True)
    preds = test_m.pop("_preds")

    print(f"\n[TEST {tag} | TTA={tta}] "
          f"Acc={test_m['accuracy']:.4f} F1={test_m['f1']:.4f} AUROC={test_m['auroc']:.4f}")

    with open(RESULTS_DIR / f"cross_{tag}.json", "w") as f:
        json.dump({"tag": tag, "best_val_f1": best_f1, **test_m}, f, indent=2)

    import csv
    with open(RESULTS_DIR / f"preds_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "y_true", "y_pred", "y_prob"])
        for path, yt, yp, pr in zip(preds["paths"], preds["y_true"],
                                    preds["y_pred"], preds["y_prob"]):
            w.writerow([path, yt, yp, f"{pr:.6f}"])

    return test_m