from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class IndustrialDefectDataset(Dataset):
    """
    Dataset loader for MVTec AD and VisA.

    Expected metadata columns:
        dataset
        category
        defect_type
        label
        image_path
        mask_path
        split

    The CSV files should use project-relative paths, for example:
        datasets/mvtec_anomaly_detection/bottle/train/good/000.png

    The project root is automatically detected from this file's location.
    """

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
        self.project_root = Path(__file__).resolve().parent.parent

        # Filter by train/test split if requested
        if split is not None:
            self.metadata = self.metadata[
                self.metadata["split"] == split
            ].reset_index(drop=True)

        # Keep only normal samples if requested
        if normal_only:
            self.metadata = self.metadata[
                self.metadata["label"] == 0
            ].reset_index(drop=True)

    def __len__(self):
        return len(self.metadata)

    def _resolve_path(self, path_value):
        """
        Convert a metadata path into an absolute path.

        Supports both:
        - project-relative paths
        - absolute paths
        """

        path_value = str(path_value).strip()

        if not path_value:
            return None

        path = Path(path_value)

        if path.is_absolute():
            return path

        return self.project_root / path

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        # --------------------------------------------------
        # Load image
        # --------------------------------------------------

        image_path = self._resolve_path(row["image_path"])

        if image_path is None or not image_path.exists():
            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        # --------------------------------------------------
        # Load mask
        # --------------------------------------------------

        mask_path = self._resolve_path(row["mask_path"])

        if mask_path is not None and mask_path.exists():
            mask = Image.open(mask_path).convert("L")
        else:
            # Normal images do not have defect masks.
            mask = Image.new("L", image.size, 0)

        # --------------------------------------------------
        # Apply image transform
        # --------------------------------------------------

        if self.transform is not None:
            image = self.transform(image)

        # --------------------------------------------------
        # Apply mask transform
        # --------------------------------------------------

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        # --------------------------------------------------
        # Convert mask to tensor
        # --------------------------------------------------

        if not torch.is_tensor(mask):
            mask = torch.tensor(
                list(mask.getdata()),
                dtype=torch.float32
            ).reshape(mask.height, mask.width)

        else:
            mask = mask.float()

        # Convert mask to binary
        mask = (mask > 0).float()

        # Remove single-channel dimension if present
        # [1, H, W] -> [H, W]
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