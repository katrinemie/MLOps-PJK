import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from PIL import Image
from data_loader import CatsDogsDataset, is_valid_image, get_transforms


def make_test_image(path, size=(64, 64)):
    """Hjælpefunktion: lav et lille fake-billede og gem det."""
    img = Image.new("RGB", size, color=(120, 80, 40))
    img.save(path)
    return path


def test_is_valid_image_with_valid_file(tmp_path):
    """is_valid_image skal returnere True for et gyldigt billede."""
    img_path = tmp_path / "kat.jpg"
    make_test_image(img_path)
    assert is_valid_image(img_path) is True


def test_is_valid_image_with_corrupt_file(tmp_path):
    """is_valid_image skal returnere False for en korrupt fil."""
    corrupt_path = tmp_path / "korrupt.jpg"
    corrupt_path.write_bytes(b"dette er ikke et billede")
    assert is_valid_image(corrupt_path) is False


def test_dataset_length(tmp_path):
    """CatsDogsDataset skal rapportere korrekt antal billeder."""
    paths = []
    labels = []
    for i in range(5):
        p = tmp_path / f"img_{i}.jpg"
        make_test_image(p)
        paths.append(p)
        labels.append(i % 2)  # skiftevis 0 og 1

    dataset = CatsDogsDataset(paths, labels)
    assert len(dataset) == 5


def test_dataset_returns_correct_label(tmp_path):
    """CatsDogsDataset skal returnere det rigtige label for hvert billede."""
    img_path = tmp_path / "hund.jpg"
    make_test_image(img_path)

    dataset = CatsDogsDataset([img_path], [1])  # label 1 = hund
    _, label = dataset[0]
    assert label == 1


def test_dataset_applies_transform(tmp_path):
    """CatsDogsDataset skal returnere en tensor når transform er sat."""
    img_path = tmp_path / "kat.jpg"
    make_test_image(img_path)

    transform = get_transforms(image_size=224, is_training=False)
    dataset = CatsDogsDataset([img_path], [0], transform=transform)

    image, _ = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)


def test_get_transforms_output_shape(tmp_path):
    """get_transforms skal producere en tensor med korrekt shape."""
    img_path = tmp_path / "test.jpg"
    make_test_image(img_path, size=(300, 200))

    img = Image.open(img_path).convert("RGB")
    transform = get_transforms(image_size=224, is_training=False)
    tensor = transform(img)

    assert tensor.shape == (3, 224, 224)
