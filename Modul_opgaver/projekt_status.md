# MLOps Projekt - Samlet Opgavestatus

> Sidst opdateret: 2026-03-18 (opdateret samme dag)

## Overblik

| Modul | Exercises | D-punkter (rapport) | Status |
|-------|-----------|---------------------|--------|
| **1: Introduction to MLOps** | 9/9 ✅ | 4/4 ✅ | Færdig |
| **2: Continuous ML** | 12/13 ⚠️ | 3/3 ✅ | Næsten færdig |
| **3: Scalable Training** | 6/6 ✅ | 6/6 ✅ | Færdig (DDP+AMP testet på AI-Lab) |
| **4: Scalable Inference** | 6/6 ✅ | 4/4 ✅ | Færdig |
| **5: Deployment** | 3/5 ⚠️ | 2/2 ✅ | Næsten færdig |
| **6: Monitoring** | 2/4 ⚠️ | 2/4 ⚠️ | Cost + Drift detection færdig |
| **7: Guest Lecture** | ? | 0/1 ❌ | Materiale mangler |
| **8: Post Deployment** | ? | 0/2 ❌ | Materiale mangler |
| **Total** | | **21/26 (81%)** | **✅ BESTÅR + BONUS! (Krav: 75% = 20/26)** |

---

## Modul 1: Introduction to MLOps ✅

### Exercises
| # | Opgave | Status | Fil/Implementering |
|---|--------|--------|--------------------|
| 1 | Opret GitHub repo med .gitignore | ✅ | `katrinemie/MLOps-PJK` |
| 2 | Initier base deep-learning projekt | ✅ | `src/` mappe |
| 3 | Tilføj model + training/test script | ✅ | `src/model.py`, `src/train.py` |
| 4 | Opret requirements.txt | ✅ | `requirements.txt` |
| 5 | Dokumenter koden | ✅ | Docstrings i src/ |
| 6 | Overhold PEP8 | ✅ | flake8 + pre-commit |
| 7 | Version control for data/model (DVC) | ✅ | `.dvc/config` → MinIO (ingen .dvc filer endnu) |
| 8 | Skriv konfigurationsfiler | ✅ | `configs/config.yaml` |
| 9 | Load configs og håndter hyperparametre | ✅ | YAML loading i train.py |

### D-punkter (rapport)
| D-punkt | Krav | Status |
|---------|------|--------|
| D1.1 | Introduktion til MLOps og forskel fra eksisterende paradigmer | ✅ |
| D1.2 | Beskrivelse af det valgte projekt | ✅ |
| D1.3 | Forventede udfordringer (udvikling, reproducerbarhed, monitoring, vedligeholdelse) | ✅ |
| D1.4 | Model card beskrivelse + reference til appendix | ✅ |

---

## Modul 2: Continuous ML ⚠️

### Exercises
| # | Opgave | Status | Fil/Implementering |
|---|--------|--------|--------------------|
| 1 | Opret development branch | ✅ | `development` branch |
| 2 | Setup pre-commits (flake8, API keys, filstørrelse) | ✅ | `.pre-commit-config.yaml` |
| 3 | Tilføj unit tests | ✅ | `tests/` (8 tests) |
| 4 | Setup CI/CD framework (Jenkins) | ✅ | `Jenkinsfile` (8 stages) |
| 5 | Pipeline trigger på nye commits | ✅ | Jenkins webhook |
| 6 | Automatiser Docker build + push til registry | ✅ | Stage i Jenkinsfile → `172.24.198.42:5000` |
| 7 | Automatiser model træning | ✅ | Stage i Jenkinsfile |
| 8 | Implementer lineage (MLflow) | ✅ | MLflow i train.py |
| 9 | Automatisk evaluering | ✅ | Stage i Jenkinsfile |
| 10 | Model registry hvis kriterier opfyldt | ✅ | accuracy ≥ 80% → MLflow registry |
| 11 | Deploy model + log til MLflow | ✅ | Deploy Model stage i `Jenkinsfile` (sætter model til Production + tagger med git hash) |
| 12 | Gem model card i MLflow | ✅ | `src/model_card.py` + logges som artifact i `train.py` |
| 13 | Branch protection + auto-merge | ❌ | Ikke konfigureret |

### D-punkter (rapport)
| D-punkt | Krav | Status |
|---------|------|--------|
| D2.1 | CI/CD pipeline overblik med flowchart + forklaring af hvert step + lineage | ✅ |
| D2.2 | Code coverage procent med screenshot | ✅ |
| D2.3 | Experiment tracking dashboards, metrics, sammenligninger | ✅ |

---

## Modul 3: Scalable Training ✅

### Exercises
| # | Opgave | Status | Fil/Implementering |
|---|--------|--------|--------------------|
| 1 | train_ddp.py med DDP multi-GPU | ✅ Testet | `src/train_ddp.py` + `src/train_ddp_benchmark.py` (1.56x speedup med 2 GPUs) |
| 2 | Memory optimization (AMP) | ✅ Testet | AMP i train.py (33% VRAM besparelse: 860→574 MB) |
| 3 | Skalér træning multi-node (torchrun/DeepSpeed) | ✅ Skrevet | `src/train_deepspeed.py` + `scripts/launch_multinode.sh` |
| 4 | ZeRO optimizer med forskellige stages | ✅ Configs | `configs/ds_config_zero{1,2,3}.json` |
| 5 | Inkluder mindst én optimering i pipeline | ✅ | AMP i train.py |
| 6 | Brug feature branches | ✅ | development branch |

### D-punkter (rapport)
| D-punkt | Krav | Status |
|---------|------|--------|
| D3.1 | Estimat af speedup fra parallelisering | ✅ |
| D3.2 | Scaling estimat for at halvere test loss (power-law) | ✅ |
| D3.3 | Effekt af multi-GPU parallelisering (DDP) | ✅ |
| D3.4 | Effekt af multi-node parallelisering | ✅ |
| D3.5 | Effekt af memory optimization (AMP) | ✅ |
| D3.6 | Effekt af ZeRO optimizer stages | ✅ |

---

## Modul 4: Scalable Inference ✅

### Exercises
| # | Opgave | Status | Fil/Implementering |
|---|--------|--------|--------------------|
| 1 | Post-training quantization (FP32→INT8) | ✅ | `src/quantize_benchmark.py` (FX static, 74.8% reduktion, 2.4-6.3x speedup) |
| 2 | Benchmark inference tid + accuracy | ✅ | `results/quantization_results.json` (friske tal fra AI-Lab job 313882) |
| 3 | Batch inference script | ✅ | `src/batch_benchmark.py` (peak 350.2 fps ved bs=16) |
| 4 | Pruning (graduelt, observer accuracy drop) | ✅ | `src/prune_finetune.py` (cliff ved 50%) |
| 5 | Fine-tune stærkt prunet model | ✅ | Knowledge distillation → 99.5% recovery |
| 6 | Inkluder mindst én optimering i pipeline | ✅ | Quantize-stage i `Jenkinsfile` |

### D-punkter (rapport)
| D-punkt | Krav | Status |
|---------|------|--------|
| D4.1 | Speedup af model kompression + accuracy forskel | ✅ |
| D4.2 | Speedup fra batch processing + latency/throughput balance | ✅ |
| D4.3 | Pruning vs. accuracy plot | ✅ |
| D4.4 | Effekt af fine-tuning på prunet model | ✅ |

---

## Modul 5: Deployment ⚠️

### Exercises
| # | Opgave | Status | Fil/Implementering |
|---|--------|--------|--------------------|
| 1 | Kvantiser ResNet50 ONNX FP32→UInt8 | ✅ | ONNX quantize_dynamic |
| 2 | Deploy på Samsung Galaxy telefon | ✅ | Android app demo |
| 3 | Dokumenter inference tid før/efter | ✅ | I rapport |
| 4 | Test endpoints (functional, robustness, performance, security) | ⚠️ | Dokumenteret i rapport, ikke implementeret i kode |
| 5 | Safeguard mod uønsket input/output | ⚠️ | Dokumenteret i rapport, ikke implementeret i kode |

### D-punkter (rapport)
| D-punkt | Krav | Status |
|---------|------|--------|
| D5.1 | Inference tid på telefon før/efter kvantisering + billede | ✅ |
| D5.2 | Endpoint testing + safeguarding strategi | ✅ |

---

## Modul 6: Monitoring ❌

### Exercises
| # | Opgave | Status | Fil/Implementering |
|---|--------|--------|--------------------|
| 1 | Carbontracker til carbon footprint (træning + inference) | ✅ Skrevet, ❌ utestet | `src/train.py` — CarbonTracker integreret, logger til `carbon_tracking.json` |
| 2 | Forudsigelser for total cost (årligt/per request) | ✅ | `src/cost_estimator.py` → `results/annual_cost_estimate.json` |
| 3 | Drift detection pipeline (data/concept drift) | ✅ | `src/drift_detector.py` — Kolmogorov-Smirnov test + performance monitoring |
| 4 | Monitoring framework (Prometheus + Grafana) | ❌ | Ikke startet |

### D-punkter (rapport)
| D-punkt | Krav | Status |
|---------|------|--------|
| D6.1 | Carbon footprint af træning | ❌ |
| D6.2 | Årligt CO₂-estimat for use case | ✅ |
| D6.3 | Drift detection + mitigering | ✅ |
| D6.4 | Screenshot af monitoring dashboard | ❌ |

---

## Modul 7: Guest Lecture ❌

| D-punkt | Krav | Status |
|---------|------|--------|
| D7.1 | Ukendt (materiale ikke tilgængeligt) | ❌ |

---

## Modul 8: Post Deployment ❌

| D-punkt | Krav | Status |
|---------|------|--------|
| D8.1 | Ukendt (materiale ikke tilgængeligt) | ❌ |
| D8.2 | Ukendt (materiale ikke tilgængeligt) | ❌ |

---

## Prioriteret TODO-liste

### 🔴 Kritisk (blokerer bestået — mangler D6.1-D6.4)
1. **Modul 6: Kør træning og aflæs CarbonTracker-output** — CarbonTracker er i train.py, skal køres på AI-Lab → noter CO₂-tal
2. **Modul 6: Årligt CO₂-estimat (D6.2)** — beregn baseret på carbontracker-resultater
3. **Modul 6: Drift detection (D6.3)** — implementer pipeline med Evidently AI
4. **Modul 6: Monitoring dashboard (D6.4)** — Prometheus + Grafana screenshot
5. **Skriv D6.1-D6.4 i rapport**

### 🟡 Vigtigt (forbedrer kvalitet)
6. Modul 2: Branch protection + auto-merge

### 🟢 Nice-to-have
8. Modul 5: Implementer endpoint testing i kode
9. DVC tracking af faktisk data (.dvc filer)
