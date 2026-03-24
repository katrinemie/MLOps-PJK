# Cats vs Dogs API

Flask-baseret REST API til billedklassificering af katte og hunde.

## Oversigt

API'et kører som en Docker-container og loader modellen fra MLflow. Det eksponerer tre endpoints til health check, modelinfo og prediktion.

- **Base URL (produktion):** `http://172.24.198.42:5000`
- **Docker image:** `172.24.198.42:5000/cats-vs-dogs-api:latest`
- **Port:** 5000

---

## Start API lokalt med Docker

```bash
docker run -p 5000:5000 \
  -e MLFLOW_TRACKING_URI=http://172.24.198.42:5050 \
  172.24.198.42:5000/cats-vs-dogs-api:latest
```

---

## Endpoints

### GET /health

Returnerer API'ets status og om modellen er loadet.

**Request**
```
GET /health
```

**Response 200**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

| Felt | Type | Beskrivelse |
|------|------|-------------|
| `status` | string | Altid `"healthy"` hvis serveren korer |
| `model_loaded` | bool | `true` hvis modellen er loadet korrekt |
| `device` | string | `"cuda"` eller `"cpu"` |

---

### GET /info

Returnerer metadata om den loadede model og tilgaengelige endpoints.

**Request**
```
GET /info
```

**Response 200**
```json
{
  "model_name": "cats-vs-dogs-model",
  "classes": ["Cat", "Dog"],
  "input_size": 224,
  "device": "cuda",
  "endpoints": {
    "/health": "GET - Health check",
    "/info": "GET - Model info",
    "/predict": "POST - Predict image"
  }
}
```

---

### POST /predict

Klassificerer et billede som kat eller hund.

**Request**

Content-Type: `multipart/form-data`

| Felt | Type | Pakraevet | Beskrivelse |
|------|------|-----------|-------------|
| `image` | fil | Ja | Billedfil (JPEG, PNG, osv.) |

**Eksempel med curl**
```bash
curl -X POST http://172.24.198.42:5000/predict \
  -F "image=@/sti/til/billede.jpg"
```

**Eksempel med Python**
```python
import requests

with open("billede.jpg", "rb") as f:
    response = requests.post(
        "http://172.24.198.42:5000/predict",
        files={"image": f}
    )

print(response.json())
```

**Response 200**
```json
{
  "prediction": "Cat",
  "confidence": 0.9823,
  "probabilities": {
    "Cat": 0.9823,
    "Dog": 0.0177
  }
}
```

| Felt | Type | Beskrivelse |
|------|------|-------------|
| `prediction` | string | `"Cat"` eller `"Dog"` |
| `confidence` | float | Sandsynlighed for den forudsagte klasse (0-1) |
| `probabilities` | object | Sandsynlighed for hver klasse |

**Fejlresponses**

| HTTP | Beskrivelse |
|------|-------------|
| 400 | Ingen `image`-felt i requesten, tomt filnavn, eller ugyldig billedfil |
| 500 | Serverfejl ved prediktion |

---

## Billedformatering

Inden prediktion forbehandles billedet automatisk:

1. Konverteres til RGB
2. Skaleres til 224x224 pixels
3. Normaliseres med ImageNet mean/std:
   - Mean: `[0.485, 0.456, 0.406]`
   - Std: `[0.229, 0.224, 0.225]`

---

## Modelindlaesning

Ved opstart forsager API'et at loade modellen fra MLflow i folgende raekkefolge:

1. **Production stage** - `models:/cats-vs-dogs-model/Production`
2. **Seneste registrerede version** - hvis ingen Production-model findes

MLflow tracking URI konfigureres via miljoevariablen `MLFLOW_TRACKING_URI` (standard: `http://172.24.198.42:5050`).

---

## Miljovariabler

| Variabel | Standard | Beskrivelse |
|----------|----------|-------------|
| `MLFLOW_TRACKING_URI` | `http://172.24.198.42:5050` | URL til MLflow tracking server |
| `PORT` | `5000` | Port som Flask-serveren lytter pa |
