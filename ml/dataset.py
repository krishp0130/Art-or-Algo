"""ImageFolder loaders with ViT-friendly preprocessing (224×224, ImageNet norm, augmentations)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ImageNet statistics (used by torchvision ViT pre-training)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_rgb(img):
    return img.convert("RGB")


def vit_train_transforms(
    *,
    size: int = 224,
    rotation_degrees: float = 15.0,
    hflip_p: float = 0.5,
    vflip_p: float = 0.5,
) -> transforms.Compose:
    """Augmented pipeline for training: resize/crop to ``size``, flips, rotation, ImageNet normalize."""
    return transforms.Compose(
        [
            transforms.Lambda(_to_rgb),
            transforms.RandomResizedCrop(
                size,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=hflip_p),
            transforms.RandomVerticalFlip(p=vflip_p),
            transforms.RandomRotation(degrees=rotation_degrees),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def vit_eval_transforms(*, size: int = 224) -> transforms.Compose:
    """Deterministic resize to ``size``×``size`` and ImageNet normalize (validation / inference)."""
    return transforms.Compose(
        [
            transforms.Lambda(_to_rgb),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    train_dir: Path | str,
    val_dir: Path | str,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    size: int = 224,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Return ``(train_loader, val_loader, meta)`` where ``meta`` includes ``class_to_idx`` / ``idx_to_class``."""
    train_dir = Path(train_dir)
    val_dir = Path(val_dir)
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing train directory: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing validation directory: {val_dir}")

    train_ds = datasets.ImageFolder(
        str(train_dir),
        transform=vit_train_transforms(size=size),
    )
    val_ds = datasets.ImageFolder(
        str(val_dir),
        transform=vit_eval_transforms(size=size),
    )
    if train_ds.class_to_idx != val_ds.class_to_idx:
        raise ValueError(
            "Train and val class folders must match. "
            f"train={train_ds.class_to_idx} val={val_ds.class_to_idx}"
        )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    meta: dict[str, Any] = {
        "class_to_idx": dict(train_ds.class_to_idx),
        "idx_to_class": idx_to_class,
        "num_classes": len(train_ds.classes),
        "class_names": list(train_ds.classes),
    }
    return train_loader, val_loader, meta
