# Module 1 - Documentation

## D1.1: Introduction to MLOps

MLOps (Machine Learning Operations) is a set of practices that combines machine learning, DevOps, and data engineering to reliably and efficiently deploy and maintain ML systems in production.

Traditional software development follows a well-defined cycle: write code, test, deploy, maintain. The output is deterministic — the same code produces the same behavior. MLOps differs because ML systems depend not only on code, but also on data and model artifacts. A change in training data can alter model behavior without any code change. This introduces challenges around reproducibility, versioning, and monitoring that traditional DevOps does not address.

Key differences from traditional development:

- **Versioning**: MLOps must track code, data, and models together — not just source code.
- **Testing**: Beyond unit and integration tests, ML systems require validation of data quality and model performance.
- **Monitoring**: Deployed models can degrade over time due to data drift, requiring continuous monitoring.
- **Reproducibility**: Experiments must be reproducible, requiring fixed seeds, pinned dependencies, and versioned datasets.

## D1.2: Project Description

The selected project is a **binary image classification system** that classifies images of cats and dogs. The system uses a ResNet18 model with transfer learning from ImageNet weights, fine-tuned on the Microsoft Cats vs Dogs dataset (~25,000 images).

The project serves as a practical vehicle for implementing a full MLOps pipeline, covering:

- Data versioning with DVC
- Experiment configuration via YAML files
- Training, evaluation, and inference pipelines
- Model checkpointing with bundled configuration for reproducibility

The tech stack consists of PyTorch and torchvision for the deep learning pipeline, DVC for data versioning, and scikit-learn for evaluation metrics.

## D1.3: Foreseen Challenges

**Development:**
- Managing different hardware environments (local CPU vs. cluster GPU) and ensuring the training pipeline works across both.
- Handling corrupted images in the dataset, which requires robust data loading with validation.

**Reproducibility:**
- Ensuring identical results across runs requires careful seed management, deterministic data splits, and pinned dependency versions.
- Transfer learning weights from external sources (ImageNet) may change between torchvision versions.

**Monitoring:**
- Detecting data drift in production — the model is trained on a specific distribution of pet images and may not generalize to images with different lighting, resolution, or backgrounds.
- Tracking model performance over time to identify degradation.

**Maintenance:**
- Retraining the model when new data becomes available, while preserving the ability to compare against previous versions.
- Managing the storage and versioning of large datasets (~866 MB) and model artifacts.

## D1.4: Model Card

An initial draft of the model card is provided below. This will be updated as the project progresses.

### Model Card: Cats vs Dogs Classifier

| Field | Description |
|-------|-------------|
| **Model name** | Cats vs Dogs Classifier |
| **Model type** | Image classification (binary) |
| **Architecture** | ResNet18 (transfer learning from ImageNet) |
| **Framework** | PyTorch |
| **Input** | RGB images, resized to 224x224 |
| **Output** | Class prediction (Cat or Dog) with probability |
| **Training data** | Microsoft Cats vs Dogs dataset (~25,000 images, 70/15/15 split) |
| **Hyperparameters** | lr=0.001, optimizer=Adam, batch_size=32, epochs=10, seed=42 |
| **Performance** | *To be completed after training* |
| **Limitations** | Trained only on cat/dog images; will not generalize to other animals. Performance may vary with image quality, resolution, and backgrounds not represented in training data. |
| **Intended use** | Educational MLOps project — not intended for production use. |
| **Ethical considerations** | No known ethical concerns for cat/dog classification. The dataset is publicly available from Microsoft Research. |
