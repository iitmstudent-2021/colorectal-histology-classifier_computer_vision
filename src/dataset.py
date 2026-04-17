"""
dataset.py  —  Data loading, augmentation, and utilities for NCT-CRC-HE-100K-NONORM.
"""

import os
from pathlib import Path

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
import torch


# ──────────────────────────────────────────────
# Class metadata
# ──────────────────────────────────────────────
CLASS_NAMES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

CLASS_DESCRIPTIONS = {
    "ADI":  "Adipose (fat) tissue",
    "BACK": "Background (non-tissue)",
    "DEB":  "Debris / cellular artifacts",
    "LYM":  "Lymphocytes (immune cells)",
    "MUC":  "Mucus",
    "MUS":  "Smooth muscle",
    "NORM": "Normal colon mucosa",
    "STR":  "Tumor-associated stroma",
    "TUM":  "Adenocarcinoma epithelium (tumor)",
}

NUM_CLASSES = len(CLASS_NAMES)

# ImageNet stats (used since we fine-tune a pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────────
# Albumentations transform pipelines
# ──────────────────────────────────────────────
def get_train_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),   # albumentations v2.x requires tuple
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # H&E stain augmentation: jitter hue/saturation aggressively
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.8),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────
# Custom Dataset
# ──────────────────────────────────────────────
class CRCDataset(Dataset):
    """
    Loads images from a folder structured as:
        root/
            ADI/  *.tif
            BACK/ *.tif
            ...
    """

    def __init__(self, root: str, transform: A.Compose = None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []   # (path, class_idx)
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

        for cls in CLASS_NAMES:
            cls_dir = self.root / cls
            if not cls_dir.exists():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(img_path), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = np.array(Image.open(path).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label


# ──────────────────────────────────────────────
# Class weights (for weighted loss)
# ──────────────────────────────────────────────
def compute_class_weights(dataset: CRCDataset) -> torch.Tensor:
    """Inverse-frequency class weights to handle mild imbalance."""
    counts = torch.zeros(NUM_CLASSES)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES   # normalise
    return weights


# ──────────────────────────────────────────────
# DataLoaders factory
# ──────────────────────────────────────────────
def build_dataloaders(
    data_root: str,
    val_split: float = 0.15,
    batch_size: int = 64,
    num_workers: int = 4,
    img_size: int = 224,
    seed: int = 42,
):
    """
    Returns (train_loader, val_loader, class_weights).
    Splits the training set by index (NOT by patient — patient IDs are not
    available in the public version of this dataset).
    """
    full_ds = CRCDataset(data_root, transform=None)   # no transform yet
    n = len(full_ds)
    n_val = int(n * val_split)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    val_idx, train_idx = indices[:n_val], indices[n_val:]

    train_samples = [full_ds.samples[i] for i in train_idx]
    val_samples   = [full_ds.samples[i] for i in val_idx]

    train_ds = CRCDataset.__new__(CRCDataset)
    train_ds.root = full_ds.root
    train_ds.class_to_idx = full_ds.class_to_idx
    train_ds.samples = train_samples
    train_ds.transform = get_train_transforms(img_size)

    val_ds = CRCDataset.__new__(CRCDataset)
    val_ds.root = full_ds.root
    val_ds.class_to_idx = full_ds.class_to_idx
    val_ds.samples = val_samples
    val_ds.transform = get_val_transforms(img_size)

    class_weights = compute_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, class_weights
