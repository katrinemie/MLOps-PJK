"""
Download Microsoft Cats vs Dogs dataset from Kaggle.

Dataset structure after download:
    data/raw/PetImages/
    ├── Cat/    (~12,500 images)
    └── Dog/    (~12,500 images)

Note: Some images in this dataset are corrupted and should be
filtered during data loading (e.g., using PIL.Image.verify()).
"""
import kagglehub
import shutil
from pathlib import Path


def download_dataset(dest: Path = Path("data/raw")) -> Path:
    """Download the cats vs dogs dataset if not already present.

    Args:
        dest: Destination directory for the dataset.

    Returns:
        Path to the downloaded dataset.
    """
    petimages_dir = dest / "PetImages"

    # Check if already downloaded (idempotent)
    if petimages_dir.exists() and any(petimages_dir.iterdir()):
        print(f"Dataset already exists at: {dest}")
        return dest

    print("Downloading dataset from Kaggle...")
    cache_path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

    dest.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, dest, dirs_exist_ok=True)

    print(f"Dataset downloaded to: {dest}")
    return dest


if __name__ == "__main__":
    download_dataset()
