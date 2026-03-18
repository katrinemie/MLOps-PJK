"""Module 3: DDP training benchmark. Measures time and VRAM for D3.3/D3.5."""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data_loader import CatsDogsDataset, SubsetWithTransform, get_transforms, load_dataset
from model import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--gpus", type=int, default=1, help="For labeling only")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    use_amp = not args.no_amp
    is_distributed = "LOCAL_RANK" in os.environ

    # Setup
    if is_distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda")

    torch.manual_seed(config["training"]["seed"] + rank)

    # Data
    data_cfg = config["data"]
    image_paths, labels = load_dataset(data_cfg["path"])
    full_ds = CatsDogsDataset(image_paths, labels, transform=None)

    total = len(full_ds)
    train_n = int(total * data_cfg["train_split"])
    val_n = int(total * data_cfg["val_split"])
    test_n = total - train_n - val_n

    gen = torch.Generator().manual_seed(config["training"]["seed"])
    train_sub, val_sub, _ = random_split(full_ds, [train_n, val_n, test_n], generator=gen)

    train_ds = SubsetWithTransform(train_sub, get_transforms(data_cfg["image_size"], True))
    val_ds = SubsetWithTransform(val_sub, get_transforms(data_cfg["image_size"], False))

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, world_size, rank, shuffle=True)
        train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                                  sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"],
                                  shuffle=True, num_workers=4, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"],
                            shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = create_model(config).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scaler = GradScaler("cuda") if use_amp else None

    # Train
    torch.cuda.reset_peak_memory_stats(device)
    train_start = time.time()

    label = f"{args.gpus}GPU_{'AMP' if use_amp else 'noAMP'}"
    if rank == 0:
        print(f"\n--- {label}: {args.epochs} epochs, bs={data_cfg['batch_size']} ---")

    for epoch in range(args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        total_loss, correct, total_samples = 0, 0, 0
        epoch_start = time.time()

        for images, lbls in train_loader:
            images, lbls = images.to(device), lbls.to(device)
            optimizer.zero_grad()

            if use_amp:
                with autocast(device_type="cuda"):
                    out = model(images)
                    loss = criterion(out, lbls)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(images)
                loss = criterion(out, lbls)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (out.argmax(1) == lbls).sum().item()
            total_samples += lbls.size(0)

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, lbls in val_loader:
                images, lbls = images.to(device), lbls.to(device)
                if use_amp:
                    with autocast(device_type="cuda"):
                        out = model(images)
                else:
                    out = model(images)
                val_correct += (out.argmax(1) == lbls).sum().item()
                val_total += lbls.size(0)

        epoch_time = time.time() - epoch_start
        if rank == 0:
            print(f"  Epoch {epoch+1}/{args.epochs}: "
                  f"train_acc={100*correct/total_samples:.1f}%, "
                  f"val_acc={100*val_correct/val_total:.1f}%, "
                  f"time={epoch_time:.1f}s")

    total_time = time.time() - train_start
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    if rank == 0:
        print(f"\n  RESULT [{label}]:")
        print(f"    Total time:  {total_time:.1f}s")
        print(f"    Per epoch:   {total_time/args.epochs:.1f}s")
        print(f"    Peak VRAM:   {peak_mb:.0f} MB")
        print(f"    Val acc:     {100*val_correct/val_total:.1f}%")

        # Save result
        os.makedirs("results", exist_ok=True)
        result = {
            "label": label,
            "gpus": args.gpus,
            "amp": use_amp,
            "epochs": args.epochs,
            "total_time_s": round(total_time, 1),
            "per_epoch_s": round(total_time / args.epochs, 1),
            "peak_vram_mb": round(peak_mb, 0),
            "val_acc": round(100 * val_correct / val_total, 1),
        }
        # Append to results file
        results_file = "results/module3_results.json"
        existing = []
        if os.path.exists(results_file):
            with open(results_file) as f:
                existing = json.load(f)
        existing.append(result)
        with open(results_file, "w") as f:
            json.dump(existing, f, indent=2)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
