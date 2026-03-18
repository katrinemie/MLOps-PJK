# Dokumentation - Lektion 1

## D1.1 - Introduktion til MLOps

MLOps (Machine Learning Operations) er en disciplin, der kombinerer Machine Learning, softwareudvikling og DevOps-principper med det formål at gøre ML-systemer reproducerbare, skalerbare og nemme at vedligeholde i produktion.

Traditionel softwareudvikling handler primært om at versionere og deployere kode. MLOps adskiller sig ved, at der er tre artefakter, der skal styres og versioneres:

- **Kode** – selve modellen og træningslogikken
- **Data** – inputdata til træning og evaluering
- **Modeller** – de trænede modelvægte

I traditionel udvikling er output af et build deterministisk: samme kode giver altid samme output. I ML er det ikke tilfældet – samme kode med forskelligt data eller tilfældig initialisering giver et andet resultat. Det stiller krav til reproducerbarhed og eksperimentsporing, som ikke findes i klassisk DevOps.

MLOps formålet er at:
- Automatisere og reproducere hele ML-livscyklussen (træning → evaluering → deployment)
- Sikre kontinuerlig integration og levering af nye modelversioner (CI/CD)
- Muliggøre overvågning og vedligeholdelse af modeller i produktion

---

## D1.2 - Projektbeskrivelse

Projektet er en binær billedklassifikation: **Cats vs. Dogs**. Modellen skal givet et billede afgøre, om det viser en kat eller en hund.

**Dataset:** Microsoft Cats vs. Dogs (~25.000 billeder fordelt ligeligt på to klasser). Data hentes fra Kaggle og lagres i MinIO på AAU MLOps-klusteret via DVC.

**Model:** ResNet18 med transfer learning fra ImageNet. Det fuldt forbundne lag er erstattet med et lineært lag med 2 outputs (kat/hund).

**Teknologier:**
| Komponent | Teknologi |
|-----------|-----------|
| Model | PyTorch / ResNet18 |
| Data versionering | DVC + MinIO (S3) |
| Eksperiment tracking | MLFlow |
| CI/CD | Jenkins |
| Container | Docker |
| Konfiguration | YAML (`configs/config.yaml`) |

**Hyperparametre (fra `configs/config.yaml`):**
- Optimizer: Adam, lr = 0.001
- Batch size: 32
- Epochs: 10
- Image size: 224×224
- Train/val/test split: 70% / 15% / 15%

---

## D1.3 - Forventede udfordringer

### Udvikling
- ResNet18 er en relativt stor model til en binær klassifikationsopgave – der kan være risiko for overfitting, særligt fordi datasættet ikke er enormt.
- Billederne i datasættet har varierende kvalitet og størrelse, hvilket kræver robust preprocessing og augmentering.

### Reproducerbarhed
- Tilfældig opdeling af data i train/val/test-splits skal fikseres med en seed-værdi, ellers vil resultater variere mellem kørsler.
- DVC bruges til at versionere data, men det kræver at alle gruppemedlemmer konfigurerer DVC-credentials korrekt lokalt.
- PyTorch's ikke-deterministiske GPU-operationer kan give marginalt forskellige resultater på tværs af hardware og CUDA-versioner.

### Overvågning
- Modellen er trænet på fotografier af hunde og katte fra Kaggle – i produktion kan der opstå **data drift**, hvis indkommende billeder er taget under andre forhold (belysning, vinkel, baggrund).
- Der er i øjeblikket ingen overvågning af modelkvalitet i produktion (concept drift detection).

### Vedligeholdelse
- Træning på klusteret er begrænset til ~30 minutter pr. run. Fulde træningskørsler med mange epoker skal planlægges på AI-Lab.
- Modelvægte og checkpoints gemmes lokalt og i MLFlow, men der mangler en klar strategi for, hvilke modeller der er i produktion og hvilke der er archiveret.

---

## D1.4 - Model Card (udkast)

> **Note:** Dette er et indledende udkast. Model card'et opdateres løbende som kurset skrider frem og den endelige version kan ses i bilag X.

| Felt | Indhold |
|------|---------|
| **Modelnavn** | cats-vs-dogs-resnet18 |
| **Modeltype** | Convolutional Neural Network (CNN), binær klassifikation |
| **Arkitektur** | ResNet18, transfer learning fra ImageNet |
| **Udviklere** | DAKI4 Gruppe 3 |
| **Version** | 0.1 (under udvikling) |
| **Dato** | Februar 2026 |
| **Licens** | AAU intern brug |

### Tilsigtet brug
Modellen er beregnet til at klassificere billeder af hunde og katte. Den er velegnet til demonstrationsformål og som led i en MLOps-pipeline.

### Datasæt
- **Kilde:** Microsoft Cats vs. Dogs (Kaggle)
- **Størrelse:** ~25.000 billeder (ca. 12.500 pr. klasse)
- **Opdeling:** 70% træning, 15% validering, 15% test

### Begrænsninger
- Modellen er kun trænet på to klasser (kat/hund) og kan ikke klassificere andre dyr korrekt.
- Modellen er **ikke** testet på billeder med meget lav opløsning, tegninger eller billeder med flere dyr.
- Der er ikke lavet en analyse af bias i datasættet.

### Metrikker (foreløbige)
*Udfyldes efter træning og evaluering er afsluttet.*

| Metrik | Værdi |
|--------|-------|
| Accuracy | TBD |
| Precision | TBD |
| Recall | TBD |
| F1 Score | TBD |
