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
| Pre-commit hooks (Flake8, API keys, filstørrelse) | ✅ `.pre-commit-config.yaml` oprettet og installeret |
| Unit tests | ✅ 8 tests, alle passed (coverage: model 85%, data_loader 45%) |
| Jenkins / CI/CD opsætning | ✅ `Jenkinsfile` oprettet — mangler Jenkins job på server |
| Pipeline trigger ved commit | ❓ Jenkinsfile klar, men job ikke opsat på `172.24.198.42:8080` endnu |
| Docker build + push til registry | ✅ `Dockerfile` oprettet |
| Automatisk træning af ny model | ✅ Indgår i Jenkinsfile |
| MLFlow / WandB lineage | ✅ MLFlow logging tilføjet i `train.py` |
| Automatisk evaluering | ✅ Indgår i Jenkinsfile |
| Model registry (MLFlow) | ✅ Indgår i Jenkinsfile (registrerer hvis accuracy ≥ 80%) |
| Deploy model + log til MLFlow | ❌ |
| Branch protection + auto-merge | ❌ |

**Modul 2: ~9/12**

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
| Modul 2 | ~75% |
| Modul 3 | 0% |
| Modul 4 | 0% |
| Modul 5 | 0% |

Næste prioritet: **Opsæt Jenkins job** på `http://172.24.198.42:8080` og kør første træning for at få MLFlow screenshots til D2.3.
