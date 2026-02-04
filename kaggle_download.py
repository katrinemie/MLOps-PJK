import kagglehub
import shutil
from pathlib import Path

path = kagglehub.dataset_download("shaunthesheep/microsoft-catsvsdogs-dataset")

dest = Path("data")
dest.mkdir(exist_ok=True)
shutil.copytree(path, dest, dirs_exist_ok=True)

print("Dataset downloaded to:", dest)
