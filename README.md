# Member 2 – Supervised Deep Learning

Binary image-level defect classification on MVTec AD and VisA.

## Architecture
- Backbone: ConvNeXt-Tiny (timm, ImageNet-pretrained)
- Loss: Focal Loss (alpha=0.25, gamma=2.0)
- Scheduler: Linear warmup + Cosine annealing
- Augmentation: flip / rotate / jitter / random erasing
- Inference: 4-view TTA
- Mixed precision (AMP)

## Experiments
1. MVTec → MVTec
2. VisA → VisA
3. MVTec → VisA
4. VisA → MVTec

## Metrics
Accuracy, Precision, Recall, F1, AUROC, AP, Confusion Matrix

## Run
```bash
python run_all.py
Outputs go to outputs/ (ignored by git).
