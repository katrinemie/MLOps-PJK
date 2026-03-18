# Dokumentation - Lektion 2

## D2.1 - Overblik over CI/CD pipeline

Den implementerede pipeline følger "Local MLOps Workflow" og trigges automatisk ved nye commits til GitHub via Jenkins.

```
git push
    │
    ▼
┌─────────────┐
│    Lint     │  Flake8 tjekker src/ for PEP8-overtrædelser
└──────┬──────┘
       │ OK
       ▼
┌─────────────┐
│    Test     │  pytest kører tests/ med coverage-rapport
└──────┬──────┘
       │ OK
       ▼
┌──────────────────────┐
│  Build & Push Docker │  Image tagges med Git commit hash
│                      │  → pushes til 172.24.198.42:5000
└──────────┬───────────┘
           │ OK
           ▼
┌─────────────┐
│  Fetch Data │  DVC henter data fra MinIO (172.24.198.42:9000)
└──────┬──────┘
       │ OK
       ▼
┌─────────────┐
│    Train    │  src/train.py køres med MLFlow tracking + AMP
│             │  Logger model card som MLFlow artifact
└──────┬──────┘
       │ OK
       ▼
┌─────────────┐
│  Evaluate   │  src/evaluate.py logger accuracy, F1, precision, recall
└──────┬──────┘
       │ OK (kun main-branch)
       ▼
┌──────────────────┐
│  Register Model  │  Registreres i MLFlow Model Registry
│  (MLFlow)        │  hvis best_val_acc ≥ 80%
└──────┬───────────┘
       │ OK (kun main-branch)
       ▼
┌──────────────────┐
│  Deploy Model    │  Seneste modelversion sættes til "Production"
│  (MLFlow)        │  i MLFlow registry + tagges med git commit hash
└──────────────────┘
```

**Forklaring af hvert trin:**

- **Lint:** Flake8 sikrer konsistent kodekvalitet og fanger simple fejl (ubrugte imports, for lange linjer osv.) inden koden når produktionsmiljøet.
- **Test:** pytest kører unit tests og genererer en coverage-rapport. Hvis tests fejler, stoppes pipelinen og koden merges ikke.
- **Build & Push Docker:** Modellen pakkes i et Docker image tagget med Git commit hash, så hver version er sporbar. Imaget pushes til det lokale registry på klusteret.
- **Fetch Data:** DVC henter den versionerede data fra MinIO, så træningen altid bruger det korrekte datasæt knyttet til det pågældende commit.
- **Train:** Modellen trænes med Automatic Mixed Precision (AMP) og metrikker logges til MLFlow for eksperiment-sporing. En model card genereres automatisk og gemmes som MLFlow artifact.
- **Evaluate:** Modellen evalueres på testdatasættet og resultater (accuracy, F1, precision, recall) logges til MLFlow.
- **Register Model:** Hvis modellen opfylder performance-kravet (`best_val_acc ≥ 80%`) registreres den i MLFlow Model Registry, så den er klar til deployment.
- **Deploy Model:** Den seneste registrerede version sættes til "Production" i MLFlow Model Registry. Versionen tagges med `deployed_by=jenkins` og det tilsvarende Git commit hash, så deploymentet er fuldt sporet.

**Lineage i pipelinen:**

Hele pipelinen er sporet via:
- **Git** – versionerer kode (commit hash bruges som Docker tag og deploy-tag)
- **DVC** – versionerer data (knytter datasæt til kode-versionen)
- **MLFlow** – tracker eksperimenter, metrikker, modelvægte, model card og deployment-status

---

## D2.2 - Code Coverage

Coverage-rapport kørt med `pytest tests/ --cov=src --cov-report=term-missing`:

| Fil | Linjer | Dækket | Coverage |
|-----|--------|--------|----------|
| `src/__init__.py` | 0 | 0 | 100% |
| `src/data_loader.py` | 82 | 37 | 45% |
| `src/model.py` | 27 | 23 | 85% |
| `src/evaluate.py` | 58 | 0 | 0% |
| `src/train.py` | 87 | 0 | 0% |
| **TOTAL** | **254** | **60** | **24%** |

**8 tests, alle passed.**

De høje coverage-tal for `model.py` (85%) og delvise coverage af `data_loader.py` (45%) skyldes, at vi tester kernefunktionaliteten (model output shape, save/load, billede-validering, transforms og dataset-klassen) uden at kræve det fulde datasæt på disk.

`train.py` og `evaluate.py` har 0% coverage da de kræver det fulde datasæt og GPU for at køre meningsfuldt — disse dækkes i stedet af integrationstest i Jenkins-pipelinen.

*Screenshot af coverage-rapport indsættes her.*

---

## D2.3 - Experiment Tracking (MLFlow)

MLFlow kører på: `http://172.24.198.42:5050`

MLFlow er integreret i `src/train.py` og logger følgende for hvert træningsrun:

**Hyperparametre (logges én gang per run):**
- Model (resnet18), pretrained, epochs, learning_rate, optimizer, batch_size, image_size, seed, amp

**Metrikker (logges per epoch):**
- `train_loss`, `train_acc`, `val_loss`, `val_acc`

**Afsluttende metrikker:**
- `best_val_acc` — bedste validerings-accuracy opnået i løbet af træningen

**Artifacts:**
- `best_model.pt` — bedste model-checkpoint
- `model_card.md` — automatisk genereret model card med arkitektur, træningskonfiguration og performance

Modeller der opfylder threshold (≥ 80% best_val_acc) registreres automatisk i MLFlow Model Registry under navnet `cats-vs-dogs-model` og sættes til "Production" ved deploy.

*Screenshots af experiment tracking dashboard, nøglemetrikker og sammenligning mellem runs indsættes her.*
