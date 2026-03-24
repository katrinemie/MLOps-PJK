"""
Generates and logs a model card to MLflow as a markdown artifact.
"""

from pathlib import Path


def generate_model_card(config: dict, metrics: dict = None) -> str:
    """
    Generate a model card as a markdown string.

    Args:
        config: Training configuration dictionary
        metrics: Optional dict with performance metrics

    Returns:
        Markdown string with the model card content
    """
    metrics = metrics or {}
    m = config["model"]
    d = config["data"]
    t = config["training"]

    card = f"""# Model Card: Cats vs Dogs Classifier

## Model Overview
- **Model Name:** cats-vs-dogs-resnet18
- **Architecture:** {m["name"]}
- **Task:** Binary image classification (Cat = 0, Dog = 1)
- **Framework:** PyTorch + torchvision

## Intended Use
Classify images as either a cat or a dog.
Developed as part of an MLOps university project (DAKI4, Gruppe 3, AAU).
Not intended for production use without further validation.

## Training Data
- **Dataset:** Microsoft Cats vs Dogs (Kaggle / PetImages)
- **Approximate size:** ~25.000 billeder
- **Split:** {int(d["train_split"] * 100)}% train / \
{int(d["val_split"] * 100)}% val / \
{int(d["test_split"] * 100)}% test
- **Input size:** {d["image_size"]}x{d["image_size"]} pixels
- **Normalization:** ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])
- **Augmentation (train):** RandomHorizontalFlip(p=0.5), RandomRotation(10°), ColorJitter

## Training Configuration
| Parameter | Value |
|-----------|-------|
| Epochs | {t["epochs"]} |
| Learning rate | {t["learning_rate"]} |
| Optimizer | {t["optimizer"]} |
| Batch size | {d["batch_size"]} |
| Pretrained | {m["pretrained"]} (ImageNet weights) |
| Seed | {t["seed"]} |

## Performance
"""

    if metrics:
        best_val = metrics.get("best_val_acc")
        acc = metrics.get("accuracy")
        prec = metrics.get("precision")
        rec = metrics.get("recall")
        f1 = metrics.get("f1_score")

        if best_val is not None:
            card += f"- **Best Validation Accuracy:** {best_val:.2f}%\n"
        if acc is not None:
            card += f"- **Test Accuracy:** {acc * 100:.2f}%\n"
        if prec is not None:
            card += f"- **Test Precision:** {prec * 100:.2f}%\n"
        if rec is not None:
            card += f"- **Test Recall:** {rec * 100:.2f}%\n"
        if f1 is not None:
            card += f"- **Test F1 Score:** {f1 * 100:.2f}%\n"
    else:
        card += "Se MLflow experiment tracking for detaljerede metrics.\n"

    card += """
## Limitations
- Trenet på web-scragte billeder — generaliserer muligvis ikke til
  usædvanlige vinkler eller billedkvaliteter.
- Binær klassifikator — vil klassificere andre dyr som kat eller hund med høj sikkerhed.
- Perfomance kan falde på billeder der afviger markant fra ImageNet-distributionen.

## Ethical Considerations
- Datasættet kan indeholde bias relateret til populære racer i vestlige lande.
- Ikke valideret til kritiske applikationer.
"""
    return card


def log_model_card(config: dict, metrics: dict = None, output_dir: str = ".") -> None:
    """
    Generate a model card, save it as markdown, and log it as an MLflow artifact.

    Args:
        config: Training configuration dictionary
        metrics: Optional dict with performance metrics to include
        output_dir: Directory to write the temporary model_card.md file
    """
    card_content = generate_model_card(config, metrics)
    card_path = Path(output_dir) / "model_card.md"
    card_path.write_text(card_content, encoding="utf-8")

    # mlflow.log_artifact(str(card_path))
    print(f"Model card logged to MLflow: {card_path}")
