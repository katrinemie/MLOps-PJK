# Projektstatus - 24. februar 2026

## Modul 1 - Basis Opsætning

| Opgave | Status |
|--------|--------|
| GitHub repo oprettet med adgang til alle | ✅ |
| `.gitignore` fil | ✅ |
| Base DL-projekt (model, train, evaluate scripts) | ✅ `src/` |
| `requirements.txt` | ✅ |
| Kode dokumentation / PEP8 | ❓ (ikke verificeret) |
| Data versionering (DVC + MinIO) | ✅ `.dvc/config` opsat |
| Konfigurationsfil (`config.yaml`) | ✅ `configs/config.yaml` |
| Hyperparametre indlæses fra config | ❓ (ikke verificeret) Men filen er lavet|

**Modul 1: ~6/8 verificeret**

---

## Modul 2 - CI/CD Pipeline

| Opgave | Status |
|--------|--------|
| Ekstra branch (development/feature) | ✅ `Peter`-branch eksisterer |
| Pre-commit hooks (Flake8, API keys, filstørrelse) | ❌ ingen `.pre-commit-config.yaml` | Peter er i gang.
| Unit tests | ❌ ingen `tests/`-mappe |
| Jenkins / CI/CD opsætning | ❌ ingen `Jenkinsfile` |
| Pipeline trigger ved commit | ❌ |
| Docker build + push til registry | ❌ ingen `Dockerfile` |
| Automatisk træning af ny model | ❌ |
| MLFlow / WandB lineage | ❌ |
| Automatisk evaluering | ❌ |
| Model registry (MLFlow) | ❌ |
| Deploy model + log til MLFlow | ❌ |
| Branch protection + auto-merge | ❌ |

**Modul 2: ~1/12 — mangler det meste**

---

## Modul 3 - Distribueret Træning

| Opgave | Status |
|--------|--------|
| `train_ddp.py` med PyTorch DDP | ❌ |
| AMP (Automatic Mixed Precision) | ❌ |
| Multi-node skalering (torchrun/DeepSpeed) | ❌ |
| ZeRO optimizer (DeepSpeed) | ❌ |

**Modul 3: 0/4**

---

## Modul 4 - Inferens Optimering

| Opgave | Status |
|--------|--------|
| Post-training kvantisering (TensorRT/PyTorch) | ❌ |
| Benchmark inferenstid + accuracy | ❌ |
| Batch inference script | ❌ |
| Pruning af model | ❌ |
| Fine-tuning af prunet model | ❌ |

**Modul 4: 0/5**

---

## Modul 5 - Android Deployment

| Opgave | Status |
|--------|--------|
| ONNX model kvantisering (FP32 → UInt8) | ❌ |
| Deploy på Android (Samsung Galaxy) | ❌ |

**Modul 5: 0/2**

---

## Samlet overblik

| Modul | Færdigt |
|-------|---------|
| Modul 1 | ~75% |
| Modul 2 | ~8% |
| Modul 3 | 0% |
| Modul 4 | 0% |
| Modul 5 | 0% |

Største prioritet: **Modul 2** — Jenkinsfile, pre-commit hooks, unit tests og MLFlow mangler alle og er fundamentet for resten af pipeline'en.
