# member2_supervised/evaluate.py
import json
import torch
from torch.utils.data import DataLoader
from config import (IMG_SIZE, BATCH_SIZE, NUM_WORKERS, DEVICE,
                    MODEL_NAME, NUM_CLASSES, DROPOUT, CKPT_DIR, RESULTS_DIR)
from dataset import DefectDataset, build_transforms
from model import DefectClassifier
from engine import evaluate


def load_model(tag_or_path):
    model = DefectClassifier(MODEL_NAME, NUM_CLASSES, pretrained=False, dropout=DROPOUT)
    ckpt = tag_or_path if str(tag_or_path).endswith(".pth") else (CKPT_DIR / f"{tag_or_path}.pth")
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


@torch.no_grad()
def evaluate_checkpoint(tag, csv_file, root_dir, split="test",
                        use_tta=True, save_preds=True):
    model = load_model(tag)
    ds = DefectDataset(csv_file, root_dir, split=split,
                       transform=build_transforms(IMG_SIZE, train=False),
                       return_path=True)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    m = evaluate(model, loader, criterion=None, use_tta=use_tta, return_preds=True)
    preds = m.pop("_preds")

    out = {"tag": tag, "split": split, "tta": use_tta, **m}
    with open(RESULTS_DIR / f"test_{tag}_{split}{'_tta' if use_tta else ''}.json", "w") as f:
        json.dump(out, f, indent=2)

    if save_preds:
        import csv
        with open(RESULTS_DIR / f"preds_{tag}_{split}{'_tta' if use_tta else ''}.csv",
                  "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "y_true", "y_pred", "y_prob"])
            for path, yt, yp, pr in zip(preds["paths"], preds["y_true"],
                                        preds["y_pred"], preds["y_prob"]):
                w.writerow([path, yt, yp, f"{pr:.6f}"])

    print(f"[{tag} | {split}{' TTA' if use_tta else ''}] "
          f"Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUROC={m['auroc']:.4f}")
    return out