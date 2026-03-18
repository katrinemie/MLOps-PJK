"""
Distributed Data Parallel (DDP) training with optional Automatic Mixed Precision (AMP).
Supports single-node multi-GPU and multi-node training via torchrun.

Usage:
    # Single node, 2 GPUs:
    torchrun --nproc_per_node=2 src/train_ddp.py --config configs/config.yaml

    # Without AMP:
    torchrun --nproc_per_node=2 src/train_ddp.py --config configs/config.yaml --no-amp

    # Multi-node (3 nodes, 1 GPU each):
    torchrun --nnodes=3 --nproc_per_node=1 \
        --master_addr=<MASTER_IP> --master_port=29500 \
        --node_rank=$NODE_RANK \
        src/train_ddp.py --config configs/config.yaml
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from carbontracker.tracker import CarbonTracker

from data_loader import (
    CatsDogsDataset,
    SubsetWithTransform,
    get_transforms,
    load_dataset,
)
from model import create_model, save_model


def setup_distributed():
    """Initialize the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def create_distributed_loaders(config, rank, world_size):
    """Create data loaders with DistributedSampler."""
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

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config["batch_size"],
        sampler=train_sampler,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config["batch_size"],
        sampler=val_sampler,
        num_workers=data_config["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler


def train_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler):
    """Train for one epoch with optional AMP."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, use_amp):
    """Validate the model with optional AMP."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            if use_amp:
                with autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train_ddp(config, use_amp=True):
    """Main DDP training function."""
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"DDP Training: {world_size} processes, AMP={'ON' if use_amp else 'OFF'}")

    torch.manual_seed(config["training"]["seed"] + rank)

    # Data
    if rank == 0:
        print("Loading data...")
    train_loader, val_loader, train_sampler = create_distributed_loaders(
        config, rank, world_size
    )

    # Model
    model = create_model(config).to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    if config["training"]["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            momentum=0.9,
        )

    scaler = GradScaler() if use_amp else None

    model_dir = Path(config["output"]["model_dir"])
    if rank == 0:
        model_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    num_epochs = config["training"]["epochs"]

    if rank == 0:
        print(f"\nStarting training for {num_epochs} epochs...")

        # Initialize carbon tracker (on rank 0 only)
        tracker = CarbonTracker(
            epochs=num_epochs,
            save_file_path=str(model_dir / "carbon_tracking.json")
        )

    # Reset peak memory for benchmarking
    torch.cuda.reset_peak_memory_stats(device)
    train_start = time.time()

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)

        if rank == 0:
            tracker.epoch_start()

        epoch_start = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, use_amp, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, use_amp)
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
                save_model(model.module, model_dir / "best_model.pt", config)

    total_time = time.time() - train_start
    peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    if rank == 0:
        save_model(model.module, model_dir / "final_model.pt", config)
        tracker.stop()
        print("\nTraining complete!")
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
        print(f"  Total training time: {total_time:.1f}s")
        print(f"  Peak GPU memory (rank 0): {peak_memory_mb:.0f} MB")
        print(f"  World size: {world_size}")
        print(f"  AMP: {'enabled' if use_amp else 'disabled'}")

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="DDP Training for Cats vs Dogs")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable Automatic Mixed Precision",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train_ddp(config, use_amp=not args.no_amp)


if __name__ == "__main__":
    main()
