"""
Model definition for Cats vs Dogs classification.
Uses ResNet18 with transfer learning from ImageNet.
"""

import torch
import torch.nn as nn
from torchvision import models


def create_model(config: dict) -> nn.Module:
    """
    Create a ResNet18 model for binary classification.

    Args:
        config: Configuration dictionary with model settings

    Returns:
        PyTorch model ready for training
    """
    model_config = config["model"]

    if model_config["name"] == "resnet18":
        if model_config["pretrained"]:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        else:
            model = models.resnet18(weights=None)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, model_config["num_classes"])

    else:
        raise ValueError(f"Unknown model: {model_config['name']}")

    return model


def save_model(model: nn.Module, path: str, config: dict = None) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model to save
        path: Path to save the model
        config: Optional config to save with the model
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
    }
    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, config: dict = None) -> nn.Module:
    """
    Load model from checkpoint.

    Args:
        path: Path to the checkpoint
        config: Configuration for model creation (if not in checkpoint)

    Returns:
        Loaded PyTorch model
    """
    checkpoint = torch.load(path, map_location="cpu")

    model_config = checkpoint.get("config", config)
    if model_config is None:
        raise ValueError("No config found in checkpoint or provided")

    model = create_model(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded from {path}")
    return model
