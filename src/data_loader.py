"""
Data loader for Cats vs Dogs dataset.
Handles loading, filtering corrupt images, and train/val/test splitting.
"""

import os
from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image


class CatsDogsDataset(Dataset):
    """PyTorch Dataset for Cats vs Dogs classification."""

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def is_valid_image(image_path: Path) -> bool:
    """Check if an image file is valid and not corrupted."""
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_dataset(data_path: str) -> Tuple[List[Path], List[int]]:
    """
    Load image paths and labels from the Cats vs Dogs dataset.
    Filters out corrupted images.

    Args:
        data_path: Path to PetImages directory containing Cat/ and Dog/ folders

    Returns:
        Tuple of (image_paths, labels) where label 0=Cat, 1=Dog
    """
    data_path = Path(data_path)
    image_paths = []
    labels = []

    class_mapping = {"Cat": 0, "Dog": 1}

    for class_name, label in class_mapping.items():
        class_dir = data_path / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Directory not found: {class_dir}")

        for image_file in class_dir.iterdir():
            if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                if is_valid_image(image_file):
                    image_paths.append(image_file)
                    labels.append(label)

    print(f"Loaded {len(image_paths)} valid images")
    print(f"  Cats: {labels.count(0)}, Dogs: {labels.count(1)}")

    return image_paths, labels


def get_transforms(image_size: int, is_training: bool = True) -> transforms.Compose:
    """
    Get image transforms for training or evaluation.

    Args:
        image_size: Target image size (square)
        is_training: Whether to include data augmentation

    Returns:
        Compose transform pipeline
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def create_data_loaders(
    config: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders.

    Args:
        config: Configuration dictionary with data settings

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_config = config["data"]

    image_paths, labels = load_dataset(data_config["path"])

    train_transform = get_transforms(data_config["image_size"], is_training=True)
    eval_transform = get_transforms(data_config["image_size"], is_training=False)

    full_dataset = CatsDogsDataset(image_paths, labels, transform=None)

    total_size = len(full_dataset)
    train_size = int(total_size * data_config["train_split"])
    val_size = int(total_size * data_config["val_split"])
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = SubsetWithTransform(train_subset, train_transform)
    val_dataset = SubsetWithTransform(val_subset, eval_transform)
    test_dataset = SubsetWithTransform(test_subset, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    print(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")

    return train_loader, val_loader, test_loader


class SubsetWithTransform(Dataset):
    """Wrapper to apply transforms to a Subset."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image_path = self.subset.dataset.image_paths[self.subset.indices[idx]]
        label = self.subset.dataset.labels[self.subset.indices[idx]]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
