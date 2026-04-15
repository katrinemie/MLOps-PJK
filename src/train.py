"""
Training script for Cats vs Dogs classification.
"""

import argparse
import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
import yaml
from tqdm import tqdm
from carbontracker.tracker import CarbonTracker

from data_loader import create_data_loaders
from model import create_model, load_model, save_model
from model_card import log_model_card


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler = None,
) -> tuple:
    """Train for one epoch with optional AMP."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    progress_bar = tqdm(train_loader, desc="Training")
    for images, labels in progress_bar:
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

        progress_bar.set_postfix(
            loss=running_loss / (progress_bar.n + 1),
            acc=100.0 * correct / total,
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    val_loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train(config: dict) -> None:
    """Main training function."""
    torch.manual_seed(config["training"]["seed"])

    device = torch.device(
        config["training"]["device"]
        if torch.cuda.is_available() and config["training"]["device"] == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    print("Loading data...")
    train_loader, val_loader, _ = create_data_loaders(config)

    print("Creating model...")
    model = create_model(config)
    model = model.to(device)

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

    # AMP: enable when running on CUDA
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    print(f"Automatic Mixed Precision (AMP): {'ON' if use_amp else 'OFF'}")

    model_dir = Path(config["output"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://172.24.198.42:5050"))
    experiment_name = "cats-vs-dogs-v2"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location="s3://pkj-catdog/mlflow-artifacts",
        )
    mlflow.set_experiment(experiment_name)

    # Initialize carbon tracker
    tracker = CarbonTracker(
        epochs=config["training"]["epochs"]
    )

    with mlflow.start_run():
        # Log alle hyperparametre fra config
        mlflow.log_params({
            "model": config["model"]["name"],
            "pretrained": config["model"]["pretrained"],
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["learning_rate"],
            "optimizer": config["training"]["optimizer"],
            "batch_size": config["data"]["batch_size"],
            "image_size": config["data"]["image_size"],
            "seed": config["training"]["seed"],
            "amp": use_amp,
        })

        best_val_acc = 0.0
        print(f"\nStarting training for {config['training']['epochs']} epochs...")

        for epoch in range(config["training"]["epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

            tracker.epoch_start()

            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            tracker.epoch_end()

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Log metrikker for denne epoch
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_model(model, model_dir / "best_model.pt", config)

        save_model(model, model_dir / "final_model.pt", config)

        # Log best model to MLflow artifact store (MinIO)
        best_model = load_model(str(model_dir / "best_model.pt"), config)
        mlflow.pytorch.log_model(best_model, "model")

        # Stop carbon tracker and log results
        tracker.stop()
        carbon_file = model_dir / "carbon_tracking.json"
        if carbon_file.exists():
            import json
            with open(carbon_file) as f:
                print("\n=== Carbon Footprint ===")
                print(json.dumps(json.load(f), indent=2))
                print("=======================\n")
        # Log slutresultat
        mlflow.log_metric("best_val_acc", best_val_acc)

        # Generer og log model card som artifact
        log_model_card(
            config=config,
            metrics={"best_val_acc": best_val_acc},
            output_dir=str(model_dir),
        )

        print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    train(config)


if __name__ == "__main__":
    main()
