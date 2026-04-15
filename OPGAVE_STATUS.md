# MLOps-PJK - Komplet Opgavestatus

> **Sidst opdateret:** 2026-04-15
> **Repo:** `katrinemie/MLOps-PJK`
> **Gruppe:** 3 (Katrine, Jonas, Peter)

---

## Samlet Overblik

| Modul | Emne | Øvelser | D-punkter | Status |
|-------|------|---------|-----------|--------|
| 1 | Introduction to MLOps | 9/9 | 4/4 | ✅ Komplet |
| 2 | Continuous ML (CI/CD) | 12/13 | 3/3 | ⚠️ Næsten komplet |
| 3 | Scalable Training | 6/6 | 6/6 | ✅ Komplet |
| 4 | Scalable Inference | 6/6 | 4/4 | ✅ Komplet |
| 5 | Deployment | 2/2 | 2/2 | ✅ Komplet |
| 6 | Monitoring | 2/3 | 2/4 | ⚠️ Delvist komplet |
| 7 | Post Deployment | 0/2 | 0/2 | ❌ Ikke påbegyndt |
| **Total** | | | **21/25 (84%)** | **Over beståelsesgrænse (75%)** |

---

## Modul 1: Introduction to MLOps

### Øvelser (9/9 ✅)

| # | Øvelse | Status | Fil / Placering |
|---|--------|--------|-----------------|
| 1 | Opret GitHub repo med .gitignore | ✅ | `MLOps-PJK/` repo + `.gitignore` |
| 2 | Initier base deep-learning projekt | ✅ | `src/model.py`, `src/train.py` |
| 3 | Tilføj model fil og training/test script | ✅ | `src/model.py` (ResNet18), `src/train.py`, `src/evaluate.py` |
| 4 | Opret requirements.txt | ✅ | `requirements.txt` |
| 5 | Dokumenter koden (docstrings) | ✅ | Alle src/ filer har docstrings |
| 6 | PEP8 kodestandarder | ✅ | flake8 i pre-commit + CI |
| 7 | Version control for data/model (DVC) | ✅ | `.dvc/config` → MinIO bucket `daki4-26-gr3` |
| 8 | Konfigurationsfiler (YAML) | ✅ | `configs/config.yaml` |
| 9 | Load konfigurationer og hyperparametre | ✅ | Bruges i `src/train.py` |

### D-punkter (4/4 ✅)

| D-punkt | Krav | Status | Bevis |
|---------|------|--------|-------|
| D1.1 | Introduktion til MLOps og forskel fra eksisterende paradigmer | ✅ | Dokumenteret i rapport |
| D1.2 | Beskrivelse af valgt projekt | ✅ | Cats vs Dogs klassifikation med ResNet18 |
| D1.3 | Forventede udfordringer (udvikling, reproducerbarhed, monitoring, vedligeholdelse) | ✅ | Dokumenteret i rapport |
| D1.4 | Model card beskrivelse (draft → færdiggøres løbende) | ✅ | `src/model_card.py` genererer model card, logges til MLflow |

### Nøglefiler
- `src/model.py` — ResNet18 med transfer learning, binary classification
- `src/data_loader.py` — Cats vs Dogs dataset, augmentering, train/val/test split
- `src/train.py` — Træningsloop med checkpointing
- `src/evaluate.py` — Accuracy, precision, recall, F1, confusion matrix
- `configs/config.yaml` — Hyperparametre
- `kaggle_download.py` — Download datasæt fra Kaggle

---

## Modul 2: Continuous ML (CI/CD)

### Øvelser (12/13 ⚠️)

| # | Øvelse | Status | Fil / Placering |
|---|--------|--------|-----------------|
| 1 | Opret development branch | ✅ | `development` branch eksisterer |
| 2 | Pre-commit hooks (flake8, API keys, filstørrelse) | ✅ | `.pre-commit-config.yaml` |
| 3 | Unit tests til kernefunktionalitet | ✅ | `tests/test_model.py` (2 tests), `tests/test_data_loader.py` (6 tests) |
| 4 | CI/CD framework (Jenkins) | ✅ | `Jenkinsfile` — 9 stages |
| 5 | Pipeline trigger ved nye commits | ✅ | Jenkins webhook |
| 6 | Automatisk Docker build + push til registry | ✅ | Stage 3 i Jenkinsfile → `172.24.198.42:5000` med git hash tag |
| 7 | Automatisk model træning | ✅ | Stage 5 i Jenkinsfile |
| 8 | Lineage tracking via MLflow | ✅ | MLflow tracking URI: `172.24.198.42:5050` |
| 9 | Automatisk evaluering af trænet model | ✅ | Stage 6 i Jenkinsfile |
| 10 | Model registry hvis accuracy ≥ 80% | ✅ | Stage 8 i Jenkinsfile → MLflow Model Registry |
| 11 | Deploy model + log deployment | ✅ | Stage 9 i Jenkinsfile |
| 12 | Gem model card i MLflow som artifact | ✅ | `src/model_card.py` → MLflow artifact |
| 13 | Branch protection + auto-merge | ❌ | **IKKE konfigureret på GitHub** |

### D-punkter (3/3 ✅)

| D-punkt | Krav | Status | Bevis |
|---------|------|--------|-------|
| D2.1 | Overblik over CI/CD pipeline (flowchart) med forklaring af steps og lineage | ✅ | Jenkinsfile + dokumentation i rapport |
| D2.2 | Code coverage procent (med screenshot) | ✅ | pytest --cov i CI |
| D2.3 | Experiment tracking dashboards, metrics, sammenligninger | ✅ | MLflow dashboards |

### Jenkins Pipeline (9 stages)
```
1. Lint (flake8) → 2. Unit Tests (pytest+coverage) → 3. Build & Push Docker
→ 4. Fetch Data (DVC/MinIO) → 5. Train (MLflow) → 6. Evaluate
→ 7. Quantize → 8. Register Model (acc ≥ 80%) → 9. Deploy API
```

### Nøglefiler
- `Jenkinsfile` — Komplet 9-stage pipeline
- `.github/workflows/ci.yml` — GitHub Actions (sekundær CI: lint + test)
- `.pre-commit-config.yaml` — flake8, detect-secrets, YAML validation
- `tests/test_model.py` + `tests/test_data_loader.py` — 8 unit tests total
- `src/serve.py` — Flask REST API til inference (deployed via Docker)
- `Dockerfile.serve` — Docker image til serving

### Mangler
- **Branch protection rules** på GitHub (øvelse 13) — skal konfigureres i repo settings

---

## Modul 3: Scalable Training

### Øvelser (6/6 ✅)

| # | Øvelse | Status | Fil / Placering |
|---|--------|--------|-----------------|
| 1 | Data parallelism (DDP) på multiple GPUs | ✅ | `src/train_ddp.py` + `src/train_ddp_benchmark.py` |
| 2 | Memory optimization (AMP) | ✅ | Integreret i `src/train.py` og `src/train_ddp_benchmark.py` |
| 3 | Multi-node skalering med DeepSpeed/torchrun | ✅ | `src/train_deepspeed.py` + `scripts/launch_multinode.sh` |
| 4 | ZeRO optimizer (stage 1, 2, 3) | ✅ | `configs/ds_config_zero1.json`, `zero2.json`, `zero3.json` |
| 5 | Inkluder optimering i MLOps pipeline | ✅ | AMP aktiveret i Jenkinsfile train stage |
| 6 | Feature branches ved eksperimenter | ✅ | Branches brugt under udvikling |

### D-punkter (6/6 ✅)

| D-punkt | Krav | Status | Resultat |
|---------|------|--------|----------|
| D3.1 | Estimat af speedup fra parallelisering | ✅ | 1.55x med 2 GPUs DDP |
| D3.2 | Estimat af skalering for at halvere test loss (power-law) | ✅ | Scaling law analyse i rapport |
| D3.3 | Effekt af multi-GPU parallelisering | ✅ | 32.6s → 20.9s per epoch (2 GPUs) |
| D3.4 | Effekt af multi-node parallelisering | ✅ | DeepSpeed resultater dokumenteret |
| D3.5 | Effekt af memory optimization (AMP) | ✅ | 33% VRAM besparelse (860 → 574 MB) |
| D3.6 | ZeRO optimizer stages sammenligning | ✅ | Stage 1/2/3 VRAM besparelser dokumenteret |

### Resultater (`results/module3_results.json`)
- **Baseline (1 GPU, no AMP):** 32.6s/epoch, 860 MB VRAM
- **1 GPU + AMP:** ~33% VRAM reduktion
- **2 GPU DDP:** 20.9s/epoch → **1.55x speedup**

### Nøglefiler
- `src/train_ddp.py` — PyTorch DDP træning
- `src/train_ddp_benchmark.py` — Benchmark 1 vs 2 GPUs ± AMP
- `src/train_deepspeed.py` — DeepSpeed ZeRO integration
- `src/summarize_module3.py` — Resultatopsummering
- `scripts/run_module3_all.sh` — SLURM script (2 GPU, 48GB RAM, 2h)
- `results/module3_results.json` — Alle benchmarkresultater

---

## Modul 4: Scalable Inference

### Øvelser (6/6 ✅)

| # | Øvelse | Status | Fil / Placering |
|---|--------|--------|-----------------|
| 1 | Post-training kvantisering (FP32 → INT8) | ✅ | `src/quantize_benchmark.py` |
| 2 | Benchmark inference tid og accuracy efter komprimering | ✅ | `results/quantization_results.json` |
| 3 | Batch inference med komprimeret model | ✅ | `src/batch_benchmark.py` |
| 4 | Pruning med gradvis fjernelse af weights | ✅ | `src/prune_finetune.py` |
| 5 | Fine-tuning af pruned model (knowledge distillation) | ✅ | `src/prune_finetune.py` |
| 6 | Inkluder inference optimering i pipeline | ✅ | Stage 7 (Quantize) i Jenkinsfile |

### D-punkter (4/4 ✅)

| D-punkt | Krav | Status | Resultat |
|---------|------|--------|----------|
| D4.1 | Speedup fra model komprimering + accuracy forskel | ✅ | INT8: 74.8% størrelsesreduktion, 2.4-6.3x speedup |
| D4.2 | Speedup fra batch processing + latency/throughput balance | ✅ | Peak: 350.2 FPS ved batch size 16, mætning dokumenteret |
| D4.3 | Pruning vs. accuracy plot | ✅ | Accuracy cliff ved 50% pruning, plot genereret |
| D4.4 | Fine-tuning effekt på pruned model | ✅ | 99.5% accuracy recovery via knowledge distillation |

### Resultater
- **Kvantisering:** INT8 giver 3.2x speedup ved batch=32 (13.53ms → 45.69ms)
- **Batch inference:** Peak throughput 350.2 FPS ved batch size 16
- **Pruning:** 95% pruning → ~50% accuracy; distillation genvinder ~80% af tabet

### Nøglefiler
- `src/quantize_benchmark.py` — FP32 → INT8 statisk kvantisering
- `src/batch_benchmark.py` — Batch størrelse vs throughput/latency
- `src/prune_finetune.py` — L1 unstructured pruning + knowledge distillation
- `src/generate_figures.py` — Genererer plots til rapport
- `scripts/run_module4_all.sh` — SLURM script (1 GPU, 24GB, 30min)
- `results/quantization_results.json`, `batch_benchmark_results.json`, `pruning_results.json`

---

## Modul 5: Deployment

### Øvelser (2/2 ✅)

| # | Øvelse | Status | Detaljer |
|---|--------|--------|----------|
| 1 | Kvantiser ResNet50 ONNX model (FP32 → UInt8) og deploy på Android | ✅ | ONNX `quantize_dynamic`, Android Studio app på Samsung Galaxy |
| 2 | Dokumenter inference tid før/efter kvantisering | ✅ | Screenshots fra telefon i rapport |

**Note:** Modul 5 var en specifik deploy-opgave (Android), IKKE i MLOps-PJK repoet. Kode lavet i separat Android Studio projekt.

### D-punkter (2/2 ✅)

| D-punkt | Krav | Status | Bevis |
|---------|------|--------|-------|
| D5.1 | Inference tid på telefon før/efter kvantisering (med screenshot) | ✅ | Billede af kørende demo i rapport |
| D5.2 | Endpoint testing (functional, robustness, performance, security) + safeguarding | ✅ | Beskrevet i rapport (teoretisk/konceptuelt) |

---

## Modul 6: Monitoring

### Øvelser (2/3 ⚠️)

| # | Øvelse | Status | Fil / Placering |
|---|--------|--------|-----------------|
| 1 | CarbonTracker til CO2 footprint under træning | ⚠️ | Integreret i `src/train.py` men **IKKE testkørt** på AI-Lab |
| 2 | Drift detection pipeline (data + concept drift) | ✅ | `src/drift_detector.py` → `results/drift_detection_report.json` |
| 3 | Monitoring framework (Prometheus + Grafana) | ❌ | **IKKE implementeret** |

### D-punkter (2/4 ⚠️)

| D-punkt | Krav | Status | Detaljer |
|---------|------|--------|----------|
| D6.1 | Carbon footprint af træning | ⚠️ | CarbonTracker i koden men mangler faktisk kørsel → **Ingen resultater endnu** |
| D6.2 | Årligt CO2 estimat for model i produktion | ✅ | `src/cost_estimator.py` → 0.23 kg CO2/år, $28.46/år |
| D6.3 | Drift detection + mitigering | ✅ | KS-test for data drift, accuracy monitoring for concept drift |
| D6.4 | Screenshot af monitoring dashboard | ❌ | **Prometheus/Grafana ikke sat op** |

### Resultater
- **Årligt CO2:** 0.23 kg (≈ 1.9 km bilkørsel)
- **Årlig energi:** 2.13 kWh (52 træninger/år)
- **Årlig pris:** $28.46
- **Drift:** KS-test detekterer feature distribution shifts, accuracy monitoring med severity klassifikation

### Nøglefiler
- `src/cost_estimator.py` — Årligt CO2 og omkostningsestimat
- `src/drift_detector.py` — Data drift (KS-test) + concept drift (accuracy decline)
- `results/annual_cost_estimate.json` — Detaljerede omkostningsresultater
- `results/drift_detection_report.json` — Drift analyserapport

### Mangler
1. **D6.1:** Kør træning med CarbonTracker på AI-Lab for at få faktiske tal
2. **D6.4:** Opsæt Prometheus + Grafana dashboard (valgfrit men anbefalet)

---

## Modul 7: Post Deployment

### Øvelser (0/2 ❌)

| # | Øvelse | Status | Detaljer |
|---|--------|--------|----------|
| 1 | **Continual Learning med Experience Replay** (MNIST) | ❌ | Ikke påbegyndt |
|   | Task 1: Træn på cifre 0-4 | ❌ | |
|   | Task 2: Naiv sekventiel træning på 5-9 (catastrophic forgetting) | ❌ | |
|   | Task 3: Implementer memory buffer + experience replay | ❌ | |
| 2 | **Unlearning med Gradient Ascent** (MNIST) | ❌ | Ikke påbegyndt |
|   | Task 1: Træn classifier på fuld MNIST (0-9) | ❌ | |
|   | Task 2: Gradient ascent for at glemme klasse "7" | ❌ | |
|   | Task 3: Evaluer forgetting og retention | ❌ | |

### D-punkter (0/2 ❌)

| D-punkt | Krav | Status |
|---------|------|--------|
| D7.1 | Sammenlign naiv sekventiel træning vs. continual learning (Replay + EWC) | ❌ Ikke påbegyndt |
| D7.2 | Evaluer unlearning: glemmer klasse "7" + beholder øvrige klasser | ❌ Ikke påbegyndt |

**Note:** Disse øvelser bruger MNIST, ikke Cats vs Dogs. Kan implementeres som selvstændige scripts.

---

## Alle Filer i MLOps-PJK og Hvilket Modul de Hører Til

### src/

| Fil | Modul | Beskrivelse |
|-----|-------|-------------|
| `model.py` | 1 | ResNet18 model definition, transfer learning, save/load |
| `data_loader.py` | 1 | Dataset loading, augmentering, train/val/test split |
| `train.py` | 1+2+6 | Træningsloop, AMP, MLflow tracking, CarbonTracker |
| `evaluate.py` | 1+2 | Evaluering: accuracy, precision, recall, F1 |
| `model_card.py` | 1+2 | Model card generation → MLflow artifact |
| `serve.py` | 2 | Flask REST API til inference |
| `train_ddp.py` | 3 | DDP træning på multiple GPUs |
| `train_ddp_benchmark.py` | 3 | Benchmark 1 vs 2 GPUs ± AMP |
| `train_deepspeed.py` | 3 | DeepSpeed ZeRO integration |
| `summarize_module3.py` | 3 | Opsummering af modul 3 resultater |
| `quantize_benchmark.py` | 4 | FP32 → INT8 kvantisering + benchmark |
| `batch_benchmark.py` | 4 | Batch inference throughput/latency |
| `prune_finetune.py` | 4 | Pruning sweep + knowledge distillation |
| `generate_figures.py` | 3+4 | Genererer plots til rapporten |
| `cost_estimator.py` | 6 | Årligt CO2 + omkostningsestimat |
| `drift_detector.py` | 6 | Data drift (KS-test) + concept drift |

### CI/CD & Config

| Fil | Modul | Beskrivelse |
|-----|-------|-------------|
| `Jenkinsfile` | 2 | 9-stage CI/CD pipeline |
| `.github/workflows/ci.yml` | 2 | GitHub Actions (lint + test) |
| `.pre-commit-config.yaml` | 2 | Pre-commit hooks (flake8, secrets, YAML) |
| `Dockerfile.serve` | 2 | Docker image til Flask serving |
| `configs/config.yaml` | 1 | Hyperparametre |

### Tests

| Fil | Modul | Tests |
|-----|-------|-------|
| `tests/test_model.py` | 2 | 2 tests (output shape, save/load) |
| `tests/test_data_loader.py` | 2 | 6 tests (validation, transforms, splits) |

### Resultater

| Fil | Modul | Indhold |
|-----|-------|---------|
| `results/module3_results.json` | 3 | DDP benchmark (speedup, VRAM) |
| `results/quantization_results.json` | 4 | INT8 kvantisering resultater |
| `results/batch_benchmark_results.json` | 4 | Batch inference metrics |
| `results/pruning_results.json` | 4 | Pruning + distillation resultater |
| `results/annual_cost_estimate.json` | 6 | Årligt CO2 og cost |
| `results/drift_detection_report.json` | 6 | Drift detection rapport |

### Scripts (SLURM)

| Fil | Modul | GPU | Tid |
|-----|-------|-----|-----|
| `scripts/run_module3_all.sh` | 3 | 2 GPU | 2h |
| `scripts/run_module4_all.sh` | 4 | 1 GPU | 30min |
| `scripts/run_figures.sh` | 3+4 | 1 GPU | 10min |

### Dokumentation

| Fil | Beskrivelse |
|-----|-------------|
| `Modul_opgaver/module1_opgave.md` | Modul 1 opgavestatus |
| `Modul_opgaver/module2_opgave.md` | Modul 2 opgavestatus |
| `Modul_opgaver/module3_opgave.md` | Modul 3 opgavestatus |
| `Modul_opgaver/module4_opgave.md` | Modul 4 opgavestatus |
| `Modul_opgaver/module5_opgave.md` | Modul 5 opgavestatus |
| `Modul_opgaver/modul6_monitoring_dokumentation.md` | Modul 6 dokumentation |
| `Modul_opgaver/projekt_status.md` | Overordnet projektstatus |
| `Peter_dok/dok_lek1.md` + `dok_lek2.md` | Peters dokumentation |

---

## Hvad Mangler? (Prioriteret)

### Kritisk (påvirker beståelse)

| Prioritet | Opgave | Modul | Estimeret indsats |
|-----------|--------|-------|-------------------|
| 🔴 1 | Implementer Continual Learning (Experience Replay) på MNIST | 7 | 2-3 timer |
| 🔴 2 | Implementer Unlearning (Gradient Ascent) på MNIST | 7 | 2-3 timer |
| 🟡 3 | Kør træning med CarbonTracker på AI-Lab for D6.1 | 6 | 30 min |

### Anbefalet (forbedrer kvalitet)

| Prioritet | Opgave | Modul | Estimeret indsats |
|-----------|--------|-------|-------------------|
| 🟡 4 | Opsæt Prometheus + Grafana monitoring for D6.4 | 6 | 2-3 timer |
| 🟢 5 | Konfigurer branch protection på GitHub | 2 | 10 min |

### Status uden Modul 7
- **Nuværende D-punkter:** 21/25 = 84% ✅ (over 75% grænse)
- **Med Modul 7 komplet:** 23/25 = 92%
- **Med alt komplet:** 25/25 = 100%

---

## Teknologi-stack

| Komponent | Teknologi | Placering |
|-----------|-----------|-----------|
| ML Framework | PyTorch 2.0 + torchvision | `src/` |
| Model | ResNet18 (transfer learning) | `src/model.py` |
| Data | Cats vs Dogs (Kaggle) | `data/raw/PetImages/` |
| Distributed Training | PyTorch DDP + DeepSpeed | `src/train_ddp.py`, `src/train_deepspeed.py` |
| Experiment Tracking | MLflow | `172.24.198.42:5050` |
| Data Versioning | DVC + MinIO | `172.24.198.42:9000`, bucket: `daki4-26-gr3` |
| CI/CD | Jenkins (primær) + GitHub Actions | `Jenkinsfile`, `.github/workflows/ci.yml` |
| Container Registry | Docker | `172.24.198.42:5000` |
| API/Serving | Flask + Flask-CORS | `src/serve.py` |
| Code Quality | flake8, detect-secrets, pytest | `.pre-commit-config.yaml` |
| Compute (træning) | AAU AI-Lab (SLURM + Singularity) | `scripts/` |
| Compute (CI/CD) | AAU MLOps kluster (daki-master/gpu1/gpu2) | Jenkins workers |
