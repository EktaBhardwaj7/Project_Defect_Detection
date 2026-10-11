"""dataset.py — torch Dataset + transforms + DataLoader factory."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import BATCH_SIZE, IMG_SIZE, MEAN, NUM_WORKERS, STD


def train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class DefectDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        if 'abs_path' not in self.df.columns:
            raise KeyError("split CSV must contain an 'abs_path' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(str(row['abs_path'])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row['label'])


def make_loader(df: pd.DataFrame, transform, shuffle: bool = False,
                batch_size: int = None, device=None, num_workers: int = None):
    bs = batch_size or BATCH_SIZE
    nw = NUM_WORKERS if num_workers is None else num_workers
    pin = bool(device is not None and getattr(device, 'type', '') == 'cuda')

    ds = DefectDataset(df, transform)
    drop_last = shuffle and len(ds) > bs and (len(ds) % bs == 1)

    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=pin,
        drop_last=drop_last,
    )