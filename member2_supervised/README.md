# member2_supervised

Supervised binary defect classification on MVTec-AD and VisA.

**Self-contained:** everything here reads metadata (read-only) and writes
splits/checkpoints/metrics *inside this folder*.

## Layout
member2_supervised/
├── config.py paths + hyper-params (only file that knows about paths)
├── data_utils.py CSV loading + path resolution
├── dataset.py torch Dataset / transforms / DataLoader
├── model.py torchvision backbones + binary head
├── losses.py class-balanced CE + focal loss
├── engine.py train/eval loops + metrics
├── train.py train one (dataset, arch)
├── evaluate.py evaluate a checkpoint
├── cross_dataset.py train on A, test on B
├── run_all.py full pipeline
├── splits/ generated split CSVs
└── outputs/ checkpoints, history.json, metrics*.json, summary.csv


## Quickstart

```bash
cd /content/drive/MyDrive/IndustrialDefectProject/member2/member2_supervised

# 0) verify paths
python -c "from config import print_config; print_config()"

# 1) build splits + train everything + evaluate + cross-test
python run_all.py

# or step by step:
python run_all.py --splits-only
python train.py    --dataset mvtec --arch resnet18
python evaluate.py --run mvtec_resnet18 --split test
python cross_dataset.py --run mvtec_resnet18 --target visa

