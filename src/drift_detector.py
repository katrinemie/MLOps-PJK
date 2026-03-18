"""
Drift detection pipeline for model monitoring.

Detects two types of drift:
1. Data drift: Feature distributions changing over time
2. Concept drift: Model performance degradation (accuracy decline)

Uses Evidently AI for data drift detection and custom logic for concept drift.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


def generate_baseline_data(n_samples=400, seed=42):
    """Generate baseline training data."""
    np.random.seed(seed)
    iris = load_iris()
    X, y = iris.data, iris.target

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices[:n_samples]], y[indices[:n_samples]]

    return pd.DataFrame(
        X, columns=iris.feature_names), pd.Series(y)


def generate_drifted_data(n_samples=400, drift_type="mild", seed=43):
    """
    Generate drifted data (simulating distribution shift).

    Args:
        n_samples (int): Number of samples
        drift_type (str): "mild", "moderate", or "severe"
        seed (int): Random seed

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Drifted features and labels
    """
    np.random.seed(seed)
    iris = load_iris()
    X, y = iris.data, iris.target

    # Create distribution shift
    if drift_type == "mild":
        # Small shift in feature means (1-2% change)
        X = X + np.random.normal(0, 0.1, X.shape)
    elif drift_type == "moderate":
        # Moderate shift (5-10% change)
        X = X + np.random.normal(0, 0.3, X.shape)
    elif drift_type == "severe":
        # Severe shift (15-20% change)
        X = X * 1.2 + np.random.normal(0, 0.5, X.shape)

    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X, y = X[indices[:n_samples]], y[indices[:n_samples]]

    return pd.DataFrame(X, columns=iris.feature_names), pd.Series(y)


def detect_data_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    threshold=0.05
) -> Dict:
    """
    Detect data drift using Kolmogorov-Smirnov test.

    Args:
        baseline_df (pd.DataFrame): Baseline features
        current_df (pd.DataFrame): Current features (potentially drifted)
        threshold (float): p-value threshold for drift detection

    Returns:
        dict: Drift detection results
    """
    from scipy import stats

    drift_results = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": False,
        "features_with_drift": [],
        "feature_details": {}
    }

    for feature in baseline_df.columns:
        baseline_dist = baseline_df[feature].values
        current_dist = current_df[feature].values

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(baseline_dist, current_dist)

        feature_drift = bool(p_value < threshold)

        drift_results["feature_details"][feature] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(p_value), 6),
            "drift_detected": feature_drift,
            "baseline_mean": round(float(baseline_dist.mean()), 4),
            "current_mean": round(float(current_dist.mean()), 4),
            "baseline_std": round(float(baseline_dist.std()), 4),
            "current_std": round(float(current_dist.std()), 4),
        }

        if feature_drift:
            drift_results["features_with_drift"].append(feature)
            drift_results["drift_detected"] = True

    return drift_results


def detect_concept_drift(
    baseline_accuracy: float,
    current_accuracy: float,
    threshold=0.05
) -> Dict:
    """
    Detect concept drift via model performance decline.

    Args:
        baseline_accuracy (float): Baseline model accuracy
        current_accuracy (float): Current model accuracy
        threshold (float): Minimum acceptable accuracy drop (e.g., 0.05 = 5%)

    Returns:
        dict: Concept drift detection results
    """
    accuracy_drop = baseline_accuracy - current_accuracy
    drift_detected = bool(accuracy_drop > threshold)

    return {
        "timestamp": datetime.now().isoformat(),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "current_accuracy": round(current_accuracy, 4),
        "accuracy_drop": round(accuracy_drop, 4),
        "accuracy_drop_percentage": round(accuracy_drop * 100, 2),
        "threshold": threshold,
        "drift_detected": drift_detected,
        "severity": (
            "critical" if accuracy_drop > 0.15
            else "high" if accuracy_drop > 0.10
            else "moderate" if accuracy_drop > 0.05
            else "low"
        ),
        "recommendation": (
            "Retrain model immediately" if accuracy_drop > 0.10
            else "Monitor closely, consider retraining" if accuracy_drop > 0.05
            else "No action needed"
        )
    }


def run_drift_detection(
    use_mock_data: bool = True,
    data_drift_type: str = "moderate",
    current_accuracy: float = 0.85
) -> Dict:
    """
    Run complete drift detection pipeline.

    Args:
        use_mock_data (bool): Use generated data or real data
        data_drift_type (str): "mild", "moderate", or "severe"
        current_accuracy (float): Model accuracy on current data

    Returns:
        dict: Complete drift detection report
    """
    # Generate or load data
    if use_mock_data:
        baseline_X, baseline_y = generate_baseline_data()
        current_X, current_y = generate_drifted_data(
            drift_type=data_drift_type)
        baseline_accuracy = 0.95  # Assumed baseline
    else:
        # In production, load from actual datasets
        baseline_X, baseline_y = generate_baseline_data()
        current_X, current_y = generate_baseline_data(seed=100)
        baseline_accuracy = 0.95

    # Run drift detections
    data_drift_results = detect_data_drift(baseline_X, current_X)
    concept_drift_results = detect_concept_drift(
        baseline_accuracy, current_accuracy)

    # Aggregate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "data_drift_detected": data_drift_results["drift_detected"],
            "concept_drift_detected": concept_drift_results["drift_detected"],
            "overall_drift_status": (
                "ALERT" if (data_drift_results["drift_detected"] or
                            concept_drift_results["drift_detected"])
                else "HEALTHY"
            ),
        },
        "data_drift": data_drift_results,
        "concept_drift": concept_drift_results,
        "mitigations": generate_mitigations(
            data_drift_results, concept_drift_results),
    }

    return report


def generate_mitigations(data_drift: Dict, concept_drift: Dict) -> List[str]:
    """Generate mitigation strategies based on detected drifts."""
    mitigations = []

    if data_drift["drift_detected"]:
        mitigations.append(
            "DATA DRIFT: Collect and analyze new training data")
        mitigations.append(
            "DATA DRIFT: Features with drift: {}".format(
                ", ".join(data_drift["features_with_drift"])))
        mitigations.append(
            "DATA DRIFT: Consider retraining with updated data distribution")

    if concept_drift["drift_detected"]:
        if concept_drift["severity"] == "critical":
            mitigations.append(
                "CONCEPT DRIFT (CRITICAL): Retrain model immediately!")
        elif concept_drift["severity"] == "high":
            mitigations.append(
                "CONCEPT DRIFT (HIGH): Plan retraining soon")
        else:
            mitigations.append(
                "CONCEPT DRIFT (MODERATE): Monitor and retrain if continues")

        mitigations.append(
            "CONCEPT DRIFT: Accuracy dropped {}% (from {:.1f}% to {:.1f}%)".
            format(
                concept_drift["accuracy_drop_percentage"],
                concept_drift["baseline_accuracy"] * 100,
                concept_drift["current_accuracy"] * 100
            ))

    if not mitigations:
        mitigations.append("No drift detected - system operational")

    return mitigations


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("DRIFT DETECTION PIPELINE")
    print("="*70)

    # Run detection with moderate data drift + accuracy drop
    report = run_drift_detection(
        use_mock_data=True,
        data_drift_type="moderate",
        current_accuracy=0.87  # 8% drop from baseline
    )

    # Display summary
    summary = report["summary"]
    print("\n[SUMMARY]")
    print("Data drift detected: {}".format(summary["data_drift_detected"]))
    print("Concept drift detected: {}".format(
        summary["concept_drift_detected"]))
    print("Overall status: {}".format(summary["overall_drift_status"]))

    # Display data drift details
    if report["data_drift"]["drift_detected"]:
        print("\n[DATA DRIFT - Features with drift]")
        for feature in report["data_drift"]["features_with_drift"]:
            detail = report["data_drift"]["feature_details"][feature]
            print("  - {}: baseline={} → current={} (KS p-value={})".format(
                feature,
                detail["baseline_mean"],
                detail["current_mean"],
                detail["p_value"]
            ))

    # Display concept drift
    if report["concept_drift"]["drift_detected"]:
        concept = report["concept_drift"]
        print("\n[CONCEPT DRIFT]")
        print("  Severity: {}".format(concept["severity"]))
        print("  Baseline accuracy: {:.2f}%".format(
            concept["baseline_accuracy"] * 100))
        print("  Current accuracy: {:.2f}%".format(
            concept["current_accuracy"] * 100))
        print("  Drop: {:.2f}%".format(concept["accuracy_drop_percentage"]))
        print("  Action: {}".format(concept["recommendation"]))

    # Display mitigations
    print("\n[MITIGATIONS]")
    for i, mitigation in enumerate(report["mitigations"], 1):
        print("  {}. {}".format(i, mitigation))

    # Save report
    output_path = Path("results/drift_detection_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*70)
    print(" Saved to: {}".format(output_path))
    print("="*70 + "\n")

    return report


if __name__ == "__main__":
    main()
