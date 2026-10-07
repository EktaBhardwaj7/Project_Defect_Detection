# member2_supervised/dataset.py
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DefectDataset(Dataset):
    """
    Binary image-level classifier dataset.
    Reads team metadata CSV, remaps Windows paths to Colab/Kaggle paths.
    """
    def __init__(self, csv_file, root_dir, split=None, transform=None,
                 return_path=False):
        self.df = pd.read_csv(csv_file)
        if split is not None and "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        self.root_dir = str(root_dir)
        self.transform = transform
        self.return_path = return_path

        # Remap paths
        self.df["full_path"] = self.df["image_path"].apply(self._remap)
        self.df["mask_full_path"] = self.df["mask_path"].apply(self._remap)

        # Drop missing files
        before = len(self.df)
        self.df = self.df[self.df["full_path"].apply(os.path.exists)].reset_index(drop=True)
        if before != len(self.df):
            print(f"[DefectDataset] Dropped {before - len(self.df)} missing files "
                  f"({len(self.df)} remain) from {csv_file}")

    def _remap(self, p):
        if pd.isna(p) or p == "" or p is None:
            return ""
        p = str(p).replace("\\", "/")
        for key in ("mvtec_anomaly_detection/", "VisA_20220922/"):
            if key in p:
                rel = p.split(key, 1)[1]
                return os.path.join(self.root_dir, rel)
        return p

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = Image.open(row["full_path"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (288, 288))

        label = int(row["label"])
        if self.transform:
            img = self.transform(img)

        if self.return_path:
            return img, torch.tensor(label, dtype=torch.long), row["full_path"]
        return img, torch.tensor(label, dtype=torch.long)


def build_transforms(img_size=288, train=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    )
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])