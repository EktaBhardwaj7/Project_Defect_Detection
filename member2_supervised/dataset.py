"""Dataset utilities for binary industrial defect classification."""

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {
    "good": 0,
    "normal": 0,
    "ok": 0,
    "healthy": 0,
    "defect": 1,
    "defective": 1,
    "anomaly": 1,
    "bad": 1,
}


def resolve_image_path(path, roots: Optional[Iterable[str]] = None):
    """Resolve image paths, including old Windows paths on Linux."""

    if pd.isna(path) or not str(path).strip():
        return None

    raw = str(path).strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part]

    candidates = [Path(raw)]

    if len(raw) > 1 and raw[1] == ":":
        without_drive = raw[2:].lstrip("/")
    else:
        without_drive = raw.lstrip("/")

    candidates.append(Path(without_drive))

    roots = [Path(root).expanduser() for root in (roots or [])]

    for root in roots:
        candidates.extend([
            root / raw.lstrip("/"),
            root / without_drive,
        ])

        for index in range(len(parts)):
            candidates.append(root.joinpath(*parts[index:]))

    seen = set()

    for candidate in candidates:
        key = str(candidate)

        if key in seen:
            continue

        seen.add(key)

        if candidate.is_file():
            return str(candidate.resolve())

    return None


def normalize_label(value):
    """Return 0 for normal and 1 for defective."""

    if pd.isna(value):
        return None

    text = str(value).strip().lower()

    if text in LABEL_MAP:
        return LABEL_MAP[text]

    try:
        number = float(text)

        if number in (0.0, 1.0):
            return int(number)

    except (TypeError, ValueError):
        pass

    return None


def load_metadata(csv_path, roots=None, strict=False):
    """
    Required CSV columns:
        image_path
        label

    Optional columns such as dataset, category, defect_type,
    mask_path and split are retained.

    IMPORTANT:
    Verify VisA labels against the official annotation files
    before training. Do not infer labels from arbitrary folder names.
    """

    csv_path = Path(csv_path)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Metadata CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {"image_path", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    df["label"] = df["label"].map(normalize_label)

    invalid = int(df["label"].isna().sum())

    if invalid:
        print(f"Skipping {invalid} rows with invalid labels.")

    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype("int64")

    df["resolved_image_path"] = df["image_path"].map(
        lambda path: resolve_image_path(path, roots)
    )

    missing_paths = df["resolved_image_path"].isna()

    if missing_paths.any():
        examples = (
            df.loc[missing_paths, "image_path"]
            .head(5)
            .tolist()
        )

        message = (
            f"{int(missing_paths.sum())} images could not be resolved. "
            f"Examples: {examples}"
        )

        if strict:
            raise FileNotFoundError(message)

        print("WARNING:", message)

    df = df.loc[~missing_paths].copy()

    # Avoid duplicate images.
    df = df.drop_duplicates(
        subset=["resolved_image_path"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No usable images remain. Check labels, CSV paths and roots."
        )

    counts = df["label"].value_counts().reindex(
        [0, 1], fill_value=0
    )

    print(
        f"Validated {len(df)} images | "
        f"normal={counts[0]} | defective={counts[1]}"
    )

    return df


def make_transform(image_size=224, augment=False):
    """ImageNet preprocessing compatible with pretrained ConvNeXt."""

    operations = [
        transforms.Resize((image_size, image_size))
    ]

    if augment:
        operations.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
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

    def __init__(
        self,
        dataframe,
        image_size=224,
        augment=False,
    ):
        self.dataframe = dataframe.reset_index(
            drop=True
        ).copy()

        self.transform = make_transform(
            image_size=image_size,
            augment=augment,
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = row["resolved_image_path"]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = self.transform(image)

        except Exception as exc:
            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from exc

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long,
        )

        return image, label