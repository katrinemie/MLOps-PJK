# MLOps-PJK

MLOps pipeline for Cats vs Dogs billedklassifikation - DAKI4 Gruppe 3.

## Projekt

Binary image classifier (Cat/Dog) baseret på **ResNet18** med transfer learning (ImageNet). Bygget med PyTorch.

## Mappestruktur

```
MLOps-PJK/
├── configs/
│   └── config.yaml          # Hyperparametre og model-konfiguration
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Dataset, augmentering, train/val/test split
│   ├── model.py             # ResNet18 model definition + save/load
│   ├── train.py             # Træningsloop med checkpointing
│   └── evaluate.py          # Evaluering (accuracy, precision, recall, F1)
├── kaggle_download.py        # Download Cats vs Dogs datasæt fra Kaggle
├── requirements.txt          # Python dependencies
├── .dvc/                     # DVC konfiguration (data versionering)
├── .dvcignore
├── .gitignore
└── README.md
```

Mapper der oprettes lokalt (gitignored):
```
data/raw/PetImages/           # Datasæt (Cat/ og Dog/ mapper)
models/                       # Gemte model checkpoints
logs/                         # Træningslogs
```

## Quick Start

### 1. Klon og installer

```bash
git clone https://github.com/katrinemie/MLOps-PJK.git
cd MLOps-PJK
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Konfigurer DVC (data versionering)

DVC remote peger allerede på vores MinIO bucket. Du skal kun tilfoeje credentials lokalt:

```bash
pip install dvc dvc-s3
dvc remote modify --local minio access_key_id daki
dvc remote modify --local minio secret_access_key dakiminio
```

Herefter kan du hente versioneret data med:
```bash
dvc pull
```

### 3. Download datasaet (foerste gang)

Hvis data ikke er DVC-tracked endnu, download manuelt:

```bash
pip install kagglehub
python kaggle_download.py
```

Dette henter Microsoft Cats vs Dogs (~25.000 billeder) til `data/raw/PetImages/`.

### 4. Traen modellen

```bash
python src/train.py
```

Konfiguration styres via `configs/config.yaml` - ingen hardcoded hyperparametre.

### 5. Evaluer modellen

```bash
python src/evaluate.py
```

## Konfiguration

Al konfiguration er i `configs/config.yaml`:

| Parameter | Vaerdi | Beskrivelse |
|-----------|--------|-------------|
| `model.name` | resnet18 | Model arkitektur |
| `model.pretrained` | true | ImageNet transfer learning |
| `model.num_classes` | 2 | Cat / Dog |
| `data.batch_size` | 32 | Batch size |
| `data.image_size` | 224 | Input resolution |
| `training.epochs` | 10 | Antal epochs |
| `training.learning_rate` | 0.001 | Learning rate |
| `training.optimizer` | adam | Optimizer |
| `training.seed` | 42 | Reproducerbarhed |

## Infrastruktur - AAU MLOps Kluster

Projektet bruger AAU's MLOps kluster. Du skal vaere paa AAU-netvaerket for at tilgaa services.

### Services

| Service | Adresse | Formaal |
|---------|---------|---------|
| MinIO (S3) | `http://172.24.198.42:9001` | Data storage + DVC remote |
| Jenkins | `http://172.24.198.42:8080` | CI/CD pipeline |
| MLFlow | `http://172.24.198.42:5050` | Experiment tracking |
| Docker Registry | `http://172.24.198.42:5000` | Container images |

### DVC Remote

DVC er konfigureret til at bruge MinIO bucket `daki4-26-gr3`:

```
Remote:   minio
Bucket:   s3://daki4-26-gr3/dvc
Endpoint: http://172.24.198.42:9000
```

Credentials gemmes i `.dvc/config.local` (gitignored) - se "Konfigurer DVC" ovenfor.

### Kluster-regler

- Arbejd ALTID i virtual environments (venv/Docker) paa worker nodes
- Gem ALDRIG persistent data paa SSD - brug MinIO
- Max ~30 min traeningsruns paa klusteret - brug AI-Lab til fulde runs
- Opgrader ALDRIG CUDA eller NVIDIA drivers

### GPU Worker Nodes

| Node | Hardware |
|------|----------|
| daki-master | Intel Ultra 9 285k, 64GB DDR5, RTX Pro 4000 Ada 20GB |
| daki-gpu1 | Intel Ultra 9 285k, 64GB DDR5, RTX Pro 4000 Ada 20GB |
| daki-gpu2 | Intel Ultra 9 285k, 64GB DDR5, RTX Pro 4000 Ada 20GB |

SSH: `ssh daki@<hostname>` (password: `daki`)

## Git Workflow

```bash
# Hent seneste aendringer
git pull
dvc pull

# Arbejd paa feature
git checkout -b feature/min-feature

# Commit og push
git add .
git commit -m "Add feature X"
git push -u origin feature/min-feature

# Versionér data-aendringer
dvc add data/
dvc push
git add data.dvc .gitignore
git commit -m "Update data version"
git push
```
