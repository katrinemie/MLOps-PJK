pipeline {
    agent any

    environment {
        REGISTRY       = '172.24.198.42:5000'
        IMAGE_NAME     = 'cats-vs-dogs'
        MLFLOW_URI     = 'http://172.24.198.42:5050'
        MIN_ACCURACY   = '0.80'
    }

    stages {

        // ----------------------------------------------------------------
        // 1. LINT
        // ----------------------------------------------------------------
        stage('Lint') {
            steps {
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    pip install --quiet flake8
                    flake8 src/ --max-line-length=100 --count --statistics
                '''
            }
        }

        // ----------------------------------------------------------------
        // 2. UNIT TESTS
        // ----------------------------------------------------------------
        stage('Test') {
            steps {
                sh '''
                    . venv/bin/activate
                    pip install --quiet pytest pytest-cov
                    pip install --quiet -r requirements.txt
                    pytest tests/ \
                        --cov=src \
                        --cov-report=xml:coverage.xml \
                        --junitxml=test-results.xml \
                        -v || true
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                }
            }
        }

        // ----------------------------------------------------------------
        // 3. BUILD & PUSH DOCKER IMAGE
        // ----------------------------------------------------------------
        stage('Build & Push Docker') {
            steps {
                sh """
                    docker build \
                        -t ${REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT} \
                        -t ${REGISTRY}/${IMAGE_NAME}:latest \
                        .
                    docker push ${REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}
                    docker push ${REGISTRY}/${IMAGE_NAME}:latest
                """
            }
        }

        // ----------------------------------------------------------------
        // 4. HENT DATA (DVC)
        // ----------------------------------------------------------------
        stage('Fetch Data') {
            steps {
                sh '''
                    . venv/bin/activate
                    pip install --quiet dvc dvc-s3
                    dvc remote modify --local minio access_key_id daki
                    dvc remote modify --local minio secret_access_key dakiminio
                    dvc pull
                '''
            }
        }

        // ----------------------------------------------------------------
        // 5. TRÆNING
        // ----------------------------------------------------------------
        stage('Train') {
            steps {
                sh '''
                    . venv/bin/activate
                    MLFLOW_TRACKING_URI=${MLFLOW_URI} \
                    python src/train.py --config configs/config.yaml
                '''
            }
        }

        // ----------------------------------------------------------------
        // 6. EVALUERING
        // ----------------------------------------------------------------
        stage('Evaluate') {
            steps {
                sh '''
                    . venv/bin/activate
                    MLFLOW_TRACKING_URI=${MLFLOW_URI} \
                    python src/evaluate.py \
                        --config configs/config.yaml \
                        --model models/best_model.pt
                '''
            }
        }

        // ----------------------------------------------------------------
        // 7. QUANTIZE MODEL (FP32 -> INT8)
        // ----------------------------------------------------------------
        stage('Quantize') {
            steps {
                sh '''
                    . venv/bin/activate
                    python src/quantize_benchmark.py
                '''
            }
        }

        // ----------------------------------------------------------------
        // 8. REGISTRER MODEL I MLFLOW (kun hvis accuracy >= MIN_ACCURACY)
        // ----------------------------------------------------------------
        stage('Register Model') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    . venv/bin/activate
                    python - <<'EOF'
import mlflow
import glob, torch, yaml

mlflow.set_tracking_uri("${MLFLOW_URI}")

with open("configs/config.yaml") as f:
    import yaml; cfg = yaml.safe_load(f)

# Find seneste MLFlow run og tjek accuracy
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("cats-vs-dogs")
if experiment:
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["start_time DESC"],
        max_results=1
    )
    if runs:
        run = runs[0]
        acc = run.data.metrics.get("best_val_acc", 0.0)
        print(f"Seneste run best_val_acc: {acc:.4f}")
        if acc >= float("${MIN_ACCURACY}"):
            mlflow.register_model(
                f"runs:/{run.info.run_id}/best_model.pt",
                "cats-vs-dogs-model"
            )
            print(f"Model registreret i MLFlow (best_val_acc={acc:.4f})")
        else:
            print(f"Accuracy {acc:.4f} under threshold ${MIN_ACCURACY} - model ikke registreret")
            exit(1)
EOF
                """
            }
        }

        // ----------------------------------------------------------------
        // 8. DEPLOY MODEL (sæt seneste version til Production i MLflow)
        // ----------------------------------------------------------------
        stage('Deploy Model') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    . venv/bin/activate
                    python - <<'EOF'
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("${MLFLOW_URI}")
client = MlflowClient()

model_name = "cats-vs-dogs-model"

# Hent seneste version af den registrerede model
versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    print(f"Ingen registrerede versioner af {model_name} - afbryder deploy")
    exit(1)

latest = sorted(versions, key=lambda v: int(v.version))[-1]
version = latest.version

# Overgange seneste version til Production
client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage="Production",
    archive_existing_versions=True,
)

# Log deploy-event som MLflow tag på model-versionen
client.set_model_version_tag(
    name=model_name,
    version=version,
    key="deployed_by",
    value="jenkins",
)
client.set_model_version_tag(
    name=model_name,
    version=version,
    key="git_commit",
    value="${GIT_COMMIT}",
)

print(f"Model '{model_name}' version {version} sat til Production")
print(f"Git commit: ${GIT_COMMIT}")
EOF
                """
            }
        }

        // ----------------------------------------------------------------
        // 9. DEPLOY API (Flask server)
        // ----------------------------------------------------------------
        stage('Deploy API') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    docker build -f Dockerfile.serve \
                        -t ${REGISTRY}/cats-vs-dogs-api:${GIT_COMMIT} \
                        -t ${REGISTRY}/cats-vs-dogs-api:latest \
                        .
                    docker push ${REGISTRY}/cats-vs-dogs-api:${GIT_COMMIT}
                    docker push ${REGISTRY}/cats-vs-dogs-api:latest

                    echo " API deployed: ${REGISTRY}/cats-vs-dogs-api:latest"
                """
            }
        }

    }

    post {
        success {
            echo "Pipeline gennemfort! Image: ${REGISTRY}/${IMAGE_NAME}:${GIT_COMMIT}"
        }
        failure {
            echo "Pipeline fejlede - tjek logs ovenfor."
        }
        always {
            cleanWs()
        }
    }
}
