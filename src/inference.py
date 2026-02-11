"""Single-image inference for Cats vs Dogs classifier."""

import argparse
import torch
from PIL import Image
from data_loader import get_transforms
from model import load_model

CLASSES = ["Cat", "Dog"]


def predict(model_path: str, image_path: str) -> None:
    model = load_model(model_path)
    model.eval()
    transform = get_transforms(224, is_training=False)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = probs.argmax().item()

    print(f"{CLASSES[pred]} ({probs[pred]:.1%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--model", default="models/best_model.pt")
    args = parser.parse_args()
    predict(args.model, args.image)
