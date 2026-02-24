import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model import create_model, save_model, load_model


# En minimal config der svarer til configs/config.yaml
CONFIG = {
    "model": {
        "name": "resnet18",
        "pretrained": False,  # False så vi ikke downloader weights under test
        "num_classes": 2,
    }
}


def test_model_output_shape():
    """Modellen skal returnere 2 outputs (kat / hund) for et batch af billeder."""
    model = create_model(CONFIG)
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch af 4 billeder
    output = model(dummy_input)
    assert output.shape == (4, 2), f"Forventet shape (4, 2), fik {output.shape}"


def test_model_save_and_load(tmp_path):
    """Gem en model og indlæs den igen - weights skal være identiske."""
    model = create_model(CONFIG)
    save_path = tmp_path / "test_model.pt"

    save_model(model, save_path, CONFIG)
    loaded_model = load_model(save_path, CONFIG)

    for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(p1, p2), "Weights er ikke identiske efter load"
