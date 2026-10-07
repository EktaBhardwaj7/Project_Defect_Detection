from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class IndustrialDefectDataset(Dataset):

    def __init__(
        self,
        metadata_csv,
        transform=None,
        mask_transform=None,
        normal_only=False,
        split=None,
    ):
        self.metadata = pd.read_csv(metadata_csv).reset_index(drop=True)

        self.transform = transform
        self.mask_transform = mask_transform

        if split is not None:
            self.metadata = self.metadata[
                self.metadata["split"] == split
            ].reset_index(drop=True)

        if normal_only:
            self.metadata = self.metadata[
                self.metadata["label"] == 0
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")

        mask_path = str(row["mask_path"]).strip()

        if mask_path and Path(mask_path).exists():
            mask = Image.open(mask_path).convert("L")
        else:
            mask = Image.new("L", image.size, 0)

        # Image transform
        if self.transform is not None:
            image = self.transform(image)

        # Mask transform
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        # Convert mask to tensor
        if not torch.is_tensor(mask):
            mask = torch.tensor(
                list(mask.getdata()),
                dtype=torch.float32
            ).reshape(mask.height, mask.width)

        # Binary mask
        mask = (mask > 0).float()

        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)

        return {
            "image": image,
            "label": int(row["label"]),
            "mask": mask,
            "category": row["category"],
            "defect_type": row["defect_type"],
            "image_path": str(image_path),
        }