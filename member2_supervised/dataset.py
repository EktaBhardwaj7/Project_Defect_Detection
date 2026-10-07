from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def build_eval_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


class SupervisedDefectDataset(Dataset):
    """
    Dataset used exclusively by Member 2's supervised experiments.

    The original metadata files are never modified.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        transform=None,
    ):
        self.metadata = pd.read_csv(metadata_csv).reset_index(drop=True)

        required = {
            "image_path",
            "label",
            "category",
            "defect_type",
        }

        missing = required - set(self.metadata.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]

        image_path = Path(row["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image does not exist: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),
            "category": row["category"],
            "defect_type": row["defect_type"],
            "image_path": str(image_path),
        }