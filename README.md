# MLOps-PJK

MLOps pipeline for **Cats vs Dogs** billedklassifikation - DAKI4 Gruppe 3.

## Quick Start

```bash
# 1. Klon repo
git clone https://github.com/katrinemie/MLOps-PJK.git
cd MLOps-PJK

# 2. Opret virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac

# 3. Installer dependencies
pip install -r requirements.txt
pip install dvc dvc-s3

# 4. Konfigurer DVC credentials (VIGTIGT - dette skal alle gøre)
dvc remote modify --local minio access_key_id daki
dvc remote modify --local minio secret_access_key dakiminio

# 5. Hent data
dvc pull                     # Hvis data er DVC-tracked
python kaggle_download.py    # Eller download fra Kaggle

# 6. Træn
python src/train.py

# 7. Evaluer
python src/evaluate.py
```

## Mappestruktur

```
MLOps-PJK/
├── configs/
│   └── config.yaml           # Alle hyperparametre (YAML)
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Dataset, augmentering, splits
│   ├── model.py              # ResNet18 definition + save/load
│   ├── train.py              # Træningsloop med checkpointing
│   └── evaluate.py           # Evaluering (accuracy, F1, confusion matrix)
├── kaggle_download.py         # Download datasæt fra Kaggle
├── requirements.txt           # Python dependencies
├── .dvc/config                # DVC remote config (committet - INGEN credentials)
└── .dvc/config.local          # DVC credentials (gitignored - lav selv)
```

Lokale mapper (gitignored):
```
data/raw/PetImages/{Cat,Dog}/  # ~25.000 billeder
models/                        # Gemte checkpoints (.pt)
logs/                          # Træningslogs
```

## DVC Setup (data versionering)

DVC remote er konfigureret til MinIO på AAU-klusteret. Konfigurationen (`.dvc/config`) er committet, men **credentials er IKKE committet**.

Hvert gruppemedlem skal køre dette en gang efter clone:

```bash
dvc remote modify --local minio access_key_id daki
dvc remote modify --local minio secret_access_key dakiminio
```

Dette opretter `.dvc/config.local` som er gitignored.

### DVC kommandoer

```bash
dvc pull                       # Hent data fra MinIO
dvc push                       # Upload data til MinIO
dvc add data/                  # Track data-ændringer
git add data.dvc .gitignore    # Commit DVC metafil
```

## Konfiguration

Al konfiguration er i `configs/config.yaml`:

```yaml
model:
  name: resnet18
  pretrained: true
  num_classes: 2

data:
  path: data/raw/PetImages
  batch_size: 32
  image_size: 224

training:
  epochs: 10
  learning_rate: 0.001
  optimizer: adam
  seed: 42
```

Ændr hyperparametre her - **ikke** i koden.

## AAU MLOps Kluster

Kræver AAU-netværk. Alle services kører på `172.24.198.42` (daki-storage):

| Service | Port | URL | Formål |
|---------|------|-----|--------|
| **MinIO** | 9001 (UI) / 9000 (API) | http://172.24.198.42:9001 | S3 storage, DVC remote |
| **Jenkins** | 8080 | http://172.24.198.42:8080 | CI/CD pipeline |
| **MLFlow** | 5050 | http://172.24.198.42:5050 | Experiment tracking |
| **Docker Registry** | 5000 | http://172.24.198.42:5000 | Container images |

### Login

| Konto | Brugernavn | Password | Brug til |
|-------|------------|----------|----------|
| Admin | `daki` | `daki` | SSH, Jenkins |
| MinIO | `daki` | `dakiminio` | MinIO UI, DVC |
| Gruppe | `daki4-26-gr3` | `daki` | SSH (gruppebruger) |

### GPU Nodes

| Node | GPU | RAM |
|------|-----|-----|
| daki-master | RTX Pro 4000 Ada 20GB | 64GB DDR5 |
| daki-gpu1 | RTX Pro 4000 Ada 20GB | 64GB DDR5 |
| daki-gpu2 | RTX Pro 4000 Ada 20GB | 64GB DDR5 |

### Regler

- **Brug altid venv eller Docker** på worker nodes
- **Gem aldrig data på SSD** - brug MinIO
- **Max ~30 min** træningsruns på klusteret (brug AI-Lab til fulde runs)
- **Rør aldrig CUDA/NVIDIA drivers**

## Git Workflow

```bash
# Hent seneste
git pull && dvc pull

# Ny feature
git checkout -b feature/mit-feature
# ... arbejd ...
git add -A && git commit -m "Add feature X"
git push -u origin feature/mit-feature

# Merge til main via PR på GitHub
```
