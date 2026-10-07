# Member 2 — Supervised Industrial Defect Detection

## 1. Overview

This module implements the supervised component of the industrial defect detection study using:

- MVTec AD
- VisA

The two datasets are intentionally kept separate.

The objective is to train an image-level binary classifier that distinguishes:

- `0` — Normal
- `1` — Defective

The supervised results are then evaluated both within the same dataset and across datasets to study generalization.

---

## 2. Research Design

The supervised experiments consist of four evaluation settings:

| Training Dataset | Test Dataset | Purpose |
|---|---|---|
| MVTec AD | MVTec AD | In-domain performance |
| VisA | VisA | In-domain performance |
| MVTec AD | VisA | Cross-dataset generalization |
| VisA | MVTec AD | Cross-dataset generalization |

MVTec AD and VisA are never merged into a single training dataset.

This separation allows us to measure how well a model trained on one industrial inspection domain transfers to another.

---

## 3. Model

The supervised baseline uses a pretrained ResNet-18.

The ImageNet-pretrained backbone is used as a feature extractor/classifier, with the final classification layer replaced by a two-class output:

```text
0 → Normal
1 → Defective
```

Transfer learning is used because the datasets are relatively specialized and the pretrained visual representation provides a strong, computationally efficient baseline.

---

## 4. Input Processing

Images are converted to RGB and resized to 224 × 224.

Training images use lightweight augmentation:

- Random horizontal flip
- Small random rotation
- Color jitter

Validation and test images use deterministic preprocessing without augmentation.

Images are normalized using the ImageNet statistics:

```text
mean = [0.485, 0.456, 0.406]

std  = [0.229, 0.224, 0.225]
```

---

## 5. Class Imbalance

Industrial defect datasets contain substantially more normal images than defective images.

To reduce the effect of this imbalance, the training loss uses class weights computed from the training split.

The weighted cross-entropy objective is:

L = -Σ w_y log(p_y)

where `w_y` is the weight assigned to the ground-truth class.

The weights are calculated using the class frequencies of the training set only.

---

## 6. Files

```text
member2_supervised/
├── __init__.py
├── create_supervised_splits.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── utils.py
└── README.md
```

### `create_supervised_splits.py`

Creates independent supervised train, validation, and test CSV files from the existing metadata.

The original metadata files are not modified.

### `dataset.py`

Provides the supervised PyTorch dataset and image transformations.

### `model.py`

Defines the ResNet-18 binary classifier.

### `train.py`

Handles:

- training
- validation
- class-weighted loss
- optimizer
- learning-rate scheduling
- checkpoint selection

### `evaluate.py`

Evaluates a trained model and produces:

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Specificity
- False-negative rate
- Confusion matrix
- ROC curve
- Per-image predictions

### `utils.py`

Contains reusable utilities for:

- random seeding
- device selection
- checkpoint loading
- JSON saving
- confusion-matrix visualization
- ROC visualization

---

## 7. Creating Supervised Splits

From the project root:

```bash
python member2_supervised/create_supervised_splits.py
```

This creates:

```text
splits/
├── mvtec_supervised_train.csv
├── mvtec_supervised_val.csv
├── mvtec_supervised_test.csv
├── visa_supervised_train.csv
├── visa_supervised_val.csv
└── visa_supervised_test.csv
```

The existing unsupervised split files are not changed.

---

## 8. Training MVTec

Example:

```bash
python member2_supervised/train.py ^
    --train-csv splits/mvtec_supervised_train.csv ^
    --val-csv splits/mvtec_supervised_val.csv ^
    --output-dir outputs/supervised/mvtec
```

On Linux/macOS:

```bash
python member2_supervised/train.py \
    --train-csv splits/mvtec_supervised_train.csv \
    --val-csv splits/mvtec_supervised_val.csv \
    --output-dir outputs/supervised/mvtec
```

The best validation checkpoint is saved as:

```text
outputs/supervised/mvtec/best_model.pt
```

---

## 9. Training VisA

```bash
python member2_supervised/train.py \
    --train-csv splits/visa_supervised_train.csv \
    --val-csv splits/visa_supervised_val.csv \
    --output-dir outputs/supervised/visa
```

The resulting checkpoint is:

```text
outputs/supervised/visa/best_model.pt
```

---

## 10. Evaluating MVTec

```bash
python member2_supervised/evaluate.py \
    --test-csv splits/mvtec_supervised_test.csv \
    --checkpoint outputs/supervised/mvtec/best_model.pt \
    --dataset-name MVTec \
    --output-dir outputs/supervised/mvtec_test
```

Expected outputs:

```text
outputs/supervised/mvtec_test/
├── test_metrics.json
├── predictions.csv
├── confusion_matrix.png
└── roc_curve.png
```

---

## 11. Evaluating VisA

```bash
python member2_supervised/evaluate.py \
    --test-csv splits/visa_supervised_test.csv \
    --checkpoint outputs/supervised/visa/best_model.pt \
    --dataset-name VisA \
    --output-dir outputs/supervised/visa_test
```

---

## 12. Cross-Dataset Evaluation

The MVTec-trained model can be evaluated directly on VisA:

```bash
python member2_supervised/evaluate.py \
    --test-csv splits/visa_supervised_test.csv \
    --checkpoint outputs/supervised/mvtec/best_model.pt \
    --dataset-name VisA \
    --output-dir outputs/supervised/mvtec_to_visa
```

Similarly, the VisA-trained model can be evaluated on MVTec:

```bash
python member2_supervised/evaluate.py \
    --test-csv splits/mvtec_supervised_test.csv \
    --checkpoint outputs/supervised/visa/best_model.pt \
    --dataset-name MVTec \
    --output-dir outputs/supervised/visa_to_mvtec
```

No target-dataset training is performed during these cross-dataset tests.

---

## 13. Evaluation Metrics

### Accuracy

The proportion of correctly classified images:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

The proportion of predicted defects that are actually defective:

```text
Precision = TP / (TP + FP)
```

### Recall

The proportion of actual defects correctly detected:

```text
Recall = TP / (TP + FN)
```

Recall is particularly important because false negatives correspond to defective products being classified as normal.

### F1-score

The harmonic mean of precision and recall:

```text
F1 = 2 × Precision × Recall / (Precision + Recall)
```

### AUROC

Measures ranking quality across classification thresholds.

### False Negative Rate

```text
FNR = FN / (TP + FN)
```

### Specificity

```text
Specificity = TN / (TN + FP)
```

---

## 14. Expected Results Table

The final experimental results should be summarized as:

| Train | Test | Accuracy | Precision | Recall | F1 | AUROC | FNR |
|---|---|---:|---:|---:|---:|---:|---:|
| MVTec | MVTec | — | — | — | — | — | — |
| VisA | VisA | — | — | — | — | — | — |
| MVTec | VisA | — | — | — | — | — | — |
| VisA | MVTec | — | — | — | — | — | — |

The values should be populated only after running the experiments.

---

## 15. Reproducibility

The experiments use a fixed random seed:

```text
42
```

The seed is applied to:

- Python random
- NumPy
- PyTorch
- CUDA

The train/validation/test split files are stored separately so that subsequent experiments use the same partitions.

---

## 16. Leakage Prevention

The following rules must be maintained:

1. MVTec and VisA are never merged.
2. Test images are never used for training.
3. Test images are never used for model selection.
4. Data augmentation is applied only to training images.
5. Ground-truth defect masks are not supplied to the classifier.
6. Class weights are computed only from the training split.
7. Cross-dataset evaluation does not retrain on the target dataset.

---

## 17. Outputs

Recommended output structure:

```text
outputs/
└── supervised/
    ├── mvtec/
    │   └── best_model.pt
    │
    ├── visa/
    │   └── best_model.pt
    │
    ├── mvtec_test/
    │   ├── test_metrics.json
    │   ├── predictions.csv
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    │
    ├── visa_test/
    │   ├── test_metrics.json
    │   ├── predictions.csv
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    │
    ├── mvtec_to_visa/
    │   ├── test_metrics.json
    │   ├── predictions.csv
    │   ├── confusion_matrix.png
    │   └── roc_curve.png
    │
    └── visa_to_mvtec/
        ├── test_metrics.json
        ├── predictions.csv
        ├── confusion_matrix.png
        └── roc_curve.png
```

---

## 18. Role in the Overall Project

Member 2 provides the supervised baseline against which the unsupervised anomaly-detection approach can be compared.

The main questions addressed by this module are:

1. How accurately can labelled industrial defect data be classified using supervised learning?
2. Does supervised performance differ between MVTec AD and VisA?
3. How well does a model trained on MVTec generalize to VisA?
4. How well does a model trained on VisA generalize to MVTec?
5. How does supervised detection compare with the unsupervised approach?
6. What performance degradation occurs when the model encounters a different industrial domain?

These experiments contribute directly to the project's comparative study of supervised and unsupervised industrial defect detection.