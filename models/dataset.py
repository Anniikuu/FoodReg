"""PyTorch Dataset, transforms, and DataLoader for food image classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config.config import (
    CLASS_INDICES_PATH,
    IMAGE_SIZE,
    PROJECT_ROOT,
)

# Standard ImageNet normalization statistics expected by MobileNetV2
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """Return image preprocessing and augmentation transforms.

    Parameters
    ----------
    is_train : bool
        If True, returns moderate realistic food augmentations (rotation +-15 deg,
        horizontal flip, zoom/scale +-10%, translation +-10%, mild color jitter)
        followed by ImageNet normalization.
        If False, returns strictly deterministic resize and ImageNet normalization.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class FoodDataset(Dataset):
    """PyTorch Dataset reading from the verified Phase 4 CSV manifests."""

    def __init__(
        self,
        manifest_path: Path,
        transform: Optional[Callable] = None,
        class_indices_path: Path = CLASS_INDICES_PATH,
        root_dir: Path = PROJECT_ROOT,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.root_dir = Path(root_dir)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        # Load deterministic class-to-index mapping
        with open(class_indices_path, "r", encoding="utf-8") as f:
            indices_data = json.load(f)
        self.class_to_idx: Dict[str, int] = indices_data["class_to_idx"]
        self.idx_to_class: Dict[int, str] = {
            int(k): v for k, v in indices_data["idx_to_class"].items()
        }

        # Read manifest rows
        self.samples: List[Tuple[str, int, str]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_rel_path = row["image_path"]
                cname = row["canonical_class"]
                if cname not in self.class_to_idx:
                    raise ValueError(f"Unknown class {cname} in manifest {self.manifest_path}")
                label_idx = self.class_to_idx[cname]
                self.samples.append((img_rel_path, label_idx, cname))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_rel_path, label_idx, _ = self.samples[idx]
        full_path = self.root_dir / img_rel_path

        # Always convert to RGB (handles PNG transparency/palettes and grayscale safely)
        with Image.open(full_path) as img:
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgb_img = img.convert("RGBA").convert("RGB")
            else:
                rgb_img = img.convert("RGB")

        if self.transform is not None:
            img_tensor = self.transform(rgb_img)
        else:
            img_tensor = transforms.functional.to_tensor(rgb_img)

        return img_tensor, label_idx


def get_dataloader(
    manifest_path: Path,
    is_train: bool = True,
    batch_size: int = 32,
    shuffle: Optional[bool] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader configured for training or evaluation."""
    if shuffle is None:
        shuffle = is_train

    transform = get_transforms(is_train=is_train)
    dataset = FoodDataset(manifest_path=manifest_path, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

