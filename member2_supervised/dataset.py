
import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {
    "good": 0,
    "normal": 0,
    "ok": 0,
    "defect": 1,
    "defective": 1,
    "anomaly": 1,
    "bad": 1,
}


def resolve_image_path(path, roots=None):
    """Resolve image paths from metadata, including old Windows paths."""
    if pd.isna(path) or not str(path).strip():
        return None

    raw = str(path).strip().replace("\\", "/")
    candidates = [Path(raw)]
    parts = [p for p in raw.split("/") if p]

    # Convert paths such as C:/old/path/category/image.png.
    if len(raw) > 1 and raw[1] == ":":
        candidates.append(Path(raw[2:].lstrip("/")))

    # Try suffixes of the original path against every image root.
    roots = [Path(r) for r in (roots or [])]
    for root in roots:
        candidates.append(root / raw.lstrip("/"))
        if len(raw) > 1 and raw[1] == ":":
            candidates.append(root / raw[2:].lstrip("/"))
        for index in range(len(parts)):
            candidates.append(root.joinpath(*parts[index:]))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    return None


def _normalize_label(value):
    if pd.isna(value):
        return None

    text = str(value).strip().lower()
    if text in LABEL_MAP:
        return LABEL_MAP[text]

    try:
        number = int(float(text))
        return number if number in (0, 1) else None
    except (TypeError, ValueError):
        return None


def load_metadata(csv_path, roots=None, strict=False):
    """
    Load metadata and add resolved_image_path.

    Expected columns:
      image_path, label

    Additional columns such as dataset, category, defect_type, mask_path,
    and split are preserved.
    """
    df = pd.read_csv(csv_path)

    required = {"image_path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    df["label"] = df["label"].map(_normalize_label)
    invalid_labels = df["label"].isna().sum()
    if invalid_labels:
        print(f"Skipping {invalid_labels} rows with invalid labels.")

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)

    df["resolved_image_path"] = df["image_path"].map(
        lambda path: resolve_image_path(path, roots)
    )

    missing_paths = df["resolved_image_path"].isna()
    if missing_paths.any():
        examples = df.loc[missing_paths, "image_path"].head(5).tolist()
        message = (
            f"{int(missing_paths.sum())} images could not be found. "
            f"Example metadata paths: {examples}"
        )
        if strict:
            raise FileNotFoundError(message)
        print("WARNING:", message)

    df = df[~missing_paths].copy()
    df = df.drop_duplicates(subset=["resolved_image_path"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No usable images remain. Check your CSV and image root folders."
        )

    return df


def make_transform(image_size=224, augment=False):
    operations = [transforms.Resize((image_size, image_size))]

    if augment:
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])

    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transforms.Compose(operations)


class DefectDataset(Dataset):
    def __init__(self, dataframe, image_size=224, augment=False):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = make_transform(image_size, augment)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = row["resolved_image_path"]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image: {image_path}") from exc

        image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label
