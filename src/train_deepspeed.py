"""
DeepSpeed ZeRO training script for Cats vs Dogs classification.
Supports ZeRO Stage 1, 2, and 3 via configuration files.

Usage:
    deepspeed src/train_deepspeed.py \
        --deepspeed_config configs/ds_config_zero1.json \
        --config configs/config.yaml
"""

import argparse
import time
from pathlib import Path

import deepspeed
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, random_split
from carbontracker.tracker import CarbonTracker

from data_loader import (
    CatsDogsDataset,
    SubsetWithTransform,
    get_transforms,
    load_dataset,
)
from model import create_model, save_model


def create_loaders(config):
    """Create train and validation data loaders."""
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
    train_subset, val_subset, _ = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = SubsetWithTransform(train_subset, train_transform)
    val_dataset = SubsetWithTransform(val_subset, eval_transform)

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

    return train_loader, val_loader


def validate(model_engine, val_loader, criterion, device):
    """Validate the model."""
    model_engine.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train_deepspeed(config, ds_config_path):
    """Main DeepSpeed training function."""
    torch.manual_seed(config["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading data...")
    train_loader, val_loader = create_loaders(config)

    print("Creating model...")
    model = create_model(config)
    criterion = nn.CrossEntropyLoss()

    # DeepSpeed initialize
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config_path,
        model_parameters=model.parameters(),
    )
    device = model_engine.local_rank

    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    num_epochs = config["training"]["epochs"]
    rank = model_engine.local_rank

    print(f"DeepSpeed training with config: {ds_config_path}")
    print(f"Starting training for {num_epochs} epochs...")

    # Initialize carbon tracker (on rank 0 only)
    tracker = CarbonTracker(
        epochs=num_epochs,
        save_file_path=str(model_dir / "carbon_tracking.json")
    ) if rank == 0 else None

    torch.cuda.reset_peak_memory_stats(device)
    train_start = time.time()

    for epoch in range(num_epochs):
        if rank == 0:
            tracker.epoch_start()

        model_engine.train()
        running_loss = 0.0
        correct = 0
        total = 0

        epoch_start = time.time()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        val_loss, val_acc = validate(model_engine, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start

        if rank == 0:
            tracker.epoch_end()

        if rank == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model_engine.module, model_dir / "best_model.pt", config)

    total_time = time.time() - train_start
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if rank == 0:
        save_model(model_engine.module, model_dir / "final_model.pt", config)
        tracker.stop()
        print("\nTraining complete!")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
        print(f"  Total training time: {total_time:.1f}s")
        print(f"  Peak GPU memory: {peak_memory_mb:.0f} MB")


def main():
    parser = argparse.ArgumentParser(description="DeepSpeed Training")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to project config file",
    )
    parser.add_argument(
        "--deepspeed_config", type=str, required=True,
        help="Path to DeepSpeed config JSON",
    )
    # DeepSpeed adds its own args (--local_rank, etc.)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_deepspeed(config, args.deepspeed_config)


if __name__ == "__main__":
    main()
