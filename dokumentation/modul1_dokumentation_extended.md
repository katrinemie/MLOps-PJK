# Module 1 – Extended Documentation

This document links each Module 1 requirement to the concrete artifacts in the repo and notes any remaining gaps.

## Snapshot of the project
- Code: `src/train.py`, `src/evaluate.py`, `src/inference.py`, `src/data_loader.py`, `src/model.py`
- Entry scripts: `train.sh`, `test.sh`, `inference.sh`
- Config: `configs/config.yaml`
- Data tracking: `data/raw.dvc` (tracks `data/raw` with Cats vs Dogs dataset)
- Artifacts/logs dirs: `models/`, `logs/` (git‑ignored)
- Environment: `requirements.txt`

## How the exercise tasks are addressed
- **Git repository & access**: Git is initialised here. Add/verify GitHub remote and invite team members (not recorded in repo metadata).
- **.gitignore**: `./.gitignore` excludes data, envs, editor files, models (`*.pt`, `*.pth`), and logs.
- **Base DL project**: ResNet18 transfer‑learning classifier for Cats vs Dogs.
- **Model + train/test scripts**: Implemented in `src/train.py`, `src/evaluate.py`, callable via `train.sh` and `test.sh`. Inference script in `src/inference.py` with wrapper `inference.sh`.
- **requirements.txt**: Lists torch/torchvision, Pillow, PyYAML, tqdm, scikit‑learn.
- **Documentation**: This file plus `dokumentation/modul1_dokumentation.md` cover narrative parts; code uses docstrings and type hints.
- **Coding practices (PEP8)**: Code formatted with reasonable readability and typing; recommend running `ruff` or `black` + `isort` before commits (tooling not yet configured).
- **Data/model versioning**: `data/raw.dvc` tracks the raw dataset; models are saved under `models/` and kept out of git. Configure a DVC remote before collaboration (see “Data & model versioning”).
- **Configs & hyperparameters**: Centralised in `configs/config.yaml`; loaded in training/eval; see section below.

## Data & model versioning
- Dataset tracked via DVC: `data/raw.dvc` points to `data/raw/` (≈25k images, ~866 MB). Current `.dvc/config` is empty ⇒ add a remote, e.g.:
  - `dvc remote add -d storage s3://<bucket>/mlops-pjk/raw`
  - `dvc push`
- Model artifacts (`models/best_model.pt`, `models/final_model.pt`) are produced by training and excluded from git via `.gitignore`. To keep models versioned, either:
  - Track them in DVC (`dvc add models/best_model.pt`) and push to the same remote, or
  - Store them in your chosen object store with naming convention and checksums recorded in the report/README.

## Configuration & hyperparameters
- File: `configs/config.yaml`
  - `model`: `resnet18`, `pretrained: true`, `num_classes: 2`
  - `data`: path `data/raw/PetImages`, `batch_size: 32`, splits 70/15/15, `image_size: 224`, `num_workers: 4`
  - `training`: `epochs: 10`, `learning_rate: 0.001`, `optimizer: adam`, `device: cuda`, `seed: 42`
  - `output`: `model_dir: models/`, `log_dir: logs/`
- Override by passing another YAML via `--config` to `train.py`/`evaluate.py`, or by editing this file. The training code seeds PyTorch, ensures deterministic splits, and saves best/final checkpoints with the config bundled.

## How to run
- **Train**: `./train.sh` (uses `configs/config.yaml`).
- **Test/evaluate**: `./test.sh --model models/best_model.pt` (outputs accuracy, precision, recall, F1, confusion matrix, classification report).
- **Single-image inference**: `./inference.sh --model models/best_model.pt <path_to_image>` (prints class + probability).

## Reproducibility notes
- Seeds: Set via `config['training']['seed']` for data splits and PyTorch.
- Data integrity: `src/data_loader.py` filters corrupted images before splitting.
- Checkpoints: Saved with attached config (`save_model`) so model recreation uses identical settings.
- Dependencies: Pin via `requirements.txt`; consider locking with `pip-tools`/`uv` if stricter reproducibility is needed.

## Model card (D1.4)
- Current draft lives in `dokumentation/modul1_dokumentation.md` (section D1.4). Please add a reference from the report to the appendix where the *final* model card will reside (e.g., `appendiks/model_card.md`) and populate metrics once training is run.

## Action items to reach full compliance
1) Add/configure DVC remote and push `data/raw.dvc` (and optionally model artifacts).
2) Record actual training/evaluation metrics in the model card and cite the checkpoint path used.
3) Document the GitHub repo URL and collaborator access in this section when set.
4) Optionally add a lightweight linter/formatter config (`ruff.toml`/`pyproject.toml`) to enforce PEP8 automatically.
