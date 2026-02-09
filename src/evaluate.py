"""
Evaluation script for Cats vs Dogs classification.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm

from data_loader import create_data_loaders
from model import load_model


def evaluate(config: dict, model_path: str) -> dict:
    """
    Evaluate model on test set.

    Args:
        config: Configuration dictionary
        model_path: Path to model checkpoint

    Returns:
        Dictionary with evaluation metrics
    """
    device = torch.device(
        config["training"]["device"]
        if torch.cuda.is_available() and config["training"]["device"] == "cuda"
        else "cpu"
    )
    print(f"Using device: {device}")

    print("Loading data...")
    _, _, test_loader = create_data_loaders(config)

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, config)
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="binary")
    recall = recall_score(all_labels, all_predictions, average="binary")
    f1 = f1_score(all_labels, all_predictions, average="binary")
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(f"  Cat predicted as Cat: {conf_matrix[0][0]}")
    print(f"  Cat predicted as Dog: {conf_matrix[0][1]}")
    print(f"  Dog predicted as Cat: {conf_matrix[1][0]}")
    print(f"  Dog predicted as Dog: {conf_matrix[1][1]}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=["Cat", "Dog"]))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Cats vs Dogs classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    evaluate(config, args.model)


if __name__ == "__main__":
    main()
