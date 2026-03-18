"""
Flask API for serving Cats vs Dogs model from MLFlow.
"""
import os
import io
import torch
import mlflow
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import torchvision.transforms as transforms

app = Flask(__name__)
CORS(app)

# Global model variable
MODEL = None
DEVICE = None
CLASS_NAMES = ['Cat', 'Dog']


def load_model():
    """Load model from MLFlow."""
    global MODEL, DEVICE

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://172.24.198.42:5050")
    )

    # Prøv at hente model fra Production stage
    client = mlflow.tracking.MlflowClient()
    model_name = "cats-vs-dogs-model"

    try:
        # Hent model fra Production
        model_uri = "models:/{}/Production".format(model_name)
        MODEL = mlflow.pytorch.load_model(model_uri)
        print(" Loaded model from Production: {}".format(model_uri))
    except Exception:
        # Hvis ikke i Production, hent seneste registreret version
        print(" Model ikke i Production stage, henter seneste version...")
        try:
            versions = client.search_model_versions("name='{}'".format(model_name))
            if versions:
                latest_version = versions[0].version
                model_uri = "models:/{}/{}".format(model_name, latest_version)
                MODEL = mlflow.pytorch.load_model(model_uri)
                print("Loaded model version {}".format(latest_version))
            else:
                raise RuntimeError("Ingen modeller fundet: {}".format(model_name))
        except Exception as e:
            print("Fejl ved loading af model: {}".format(str(e)))
            raise

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    print(" Using device: {}".format(DEVICE))


def preprocess_image(image_file, image_size=224):
    """Preprocess image for model."""
    try:
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image_tensor = transform(image).unsqueeze(0)
        return image_tensor.to(DEVICE)
    except Exception as e:
        raise ValueError("Error preprocessing image: {}".format(str(e)))


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE)
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict billede class.
    Kræver: POST request med 'image' file
    """
    try:
        # Check if file is present
        if 'image' not in request.files:
            return jsonify({"error": "No 'image' field in request"}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Preprocess image
        image_tensor = preprocess_image(image_file)

        # Make prediction
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return jsonify({
            "prediction": CLASS_NAMES[predicted_class],
            "confidence": round(confidence, 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(probabilities[0][i]), 4)
                for i in range(len(CLASS_NAMES))
            }
        }), 200

    except ValueError as e:
        print("[ERROR] Validation error: {}".format(str(e)))
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print("[ERROR] Server error: {}".format(str(e)))
        return jsonify({"error": "Server error: {}".format(str(e))}), 500


@app.route('/info', methods=['GET'])
def info():
    """Get model info."""
    return jsonify({
        "model_name": "cats-vs-dogs-model",
        "classes": CLASS_NAMES,
        "input_size": 224,
        "device": str(DEVICE),
        "endpoints": {
            "/health": "GET - Health check",
            "/info": "GET - Model info",
            "/predict": "POST - Predict image"
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("[INFO] Starting Flask server...")
    load_model()
    print("[INFO] Flask server running on port 5000")
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=False,
        threaded=False
    )
