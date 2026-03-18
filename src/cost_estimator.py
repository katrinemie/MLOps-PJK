"""
Annual CO2 and cost estimator for model training.

Reads carbon tracking data and estimates:
- Annual CO2 emissions based on training frequency
- Annual cloud compute costs based on hardware usage
- Cost per inference request
"""

import json
import os
from pathlib import Path
from datetime import datetime


def load_carbon_tracking(carbon_file="results/carbon_tracking.json"):
    """
    Load carbon tracking data from file or use defaults.

    Returns:
        dict: Carbon tracking metrics with keys:
            - total_energy (kWh)
            - total_co2 (g CO2eq)
            - total_time (seconds)
            - epochs (number of epochs)
    """
    if os.path.exists(carbon_file):
        with open(carbon_file, "r") as f:
            return json.load(f)
    else:
        # Default values if training hasn't been run yet
        # Based on typical ResNet50 training on 1 GPU
        return {
            "total_energy": 0.041,  # kWh (estimated for ~1 hour training)
            "total_co2": 4.4,  # g CO2eq (based on EU average intensity ~107 gCO2/kWh)
            "total_time": 3600,  # seconds (~1 hour)
            "epochs": 10,
            "device": "GPU"
        }


def estimate_annual_costs(
    carbon_data,
    trainings_per_year=52,  # Weekly retraining
    gpu_hourly_cost=0.44,  # AWS p3.2xlarge per hour in USD
    region_co2_intensity=107,  # gCO2/kWh (EU-28 average)
):
    """
    Estimate annual costs and emissions.

    Args:
        carbon_data (dict): Output from carbontracker
        trainings_per_year (int): Expected retrainings per year (default: 52 = weekly)
        gpu_hourly_cost (float): USD per GPU-hour (default: AWS p3.2xlarge)
        region_co2_intensity (float): gCO2/kWh for region (default: EU-28 avg)

    Returns:
        dict: Annual estimates with CO2, energy, and cost metrics
    """
    # Per-training metrics
    energy_per_training_kwh = carbon_data.get("total_energy", 0.041)
    time_per_training_hours = carbon_data.get("total_time", 3600) / 3600

    # Annual scaling
    annual_energy_kwh = energy_per_training_kwh * trainings_per_year
    annual_co2_grams = annual_energy_kwh * region_co2_intensity
    annual_co2_kg = annual_co2_grams / 1000
    annual_co2_metric_tons = annual_co2_kg / 1000

    # Cost estimation
    # Training cost = hours * hourly_rate
    annual_training_cost_usd = time_per_training_hours * gpu_hourly_cost * trainings_per_year

    # Add infrastructure overhead (~20% for storage, networking, etc.)
    annual_total_cost_usd = annual_training_cost_usd * 1.2

    # Cost per inference (assume 1M inferences per year,
    # ~0.001 USD per inference on serverless)
    inferences_per_year = 1_000_000
    cost_per_inference_usd = 0.000001
    annual_inference_cost_usd = inferences_per_year * cost_per_inference_usd

    # Total annual cost
    total_annual_cost_usd = annual_total_cost_usd + annual_inference_cost_usd

    # CO2 per dollar spent
    co2_per_dollar = annual_co2_kg / total_annual_cost_usd if total_annual_cost_usd > 0 else 0

    return {
        "timestamp": datetime.now().isoformat(),
        "training": {
            "per_training_energy_kwh": round(energy_per_training_kwh, 6),
            "per_training_time_hours": round(time_per_training_hours, 2),
            "per_training_co2_grams": round(energy_per_training_kwh * region_co2_intensity, 2),
        },
        "annual": {
            "trainings": trainings_per_year,
            "total_energy_kwh": round(annual_energy_kwh, 2),
            "total_co2_grams": round(annual_co2_grams, 2),
            "total_co2_kg": round(annual_co2_kg, 2),
            "total_co2_metric_tons": round(annual_co2_metric_tons, 6),
        },
        "cost": {
            "annual_training_cost_usd": round(annual_training_cost_usd, 2),
            "annual_infrastructure_overhead_usd": round(annual_training_cost_usd * 0.2, 2),
            "annual_inference_cost_usd": round(annual_inference_cost_usd, 2),
            "total_annual_cost_usd": round(total_annual_cost_usd, 2),
            "cost_per_training_usd": round(annual_training_cost_usd / trainings_per_year, 2),
            "cost_per_inference_usd": cost_per_inference_usd,
        },
        "sustainability": {
            "co2_per_dollar_spent": round(co2_per_dollar, 6),
            "equivalent_km_car_per_year": round(annual_co2_kg / 0.12, 2),
            "equivalent_trees_to_offset_per_year": round(
                annual_co2_kg / 20, 0),  # 20 kg CO2/tree/year
        },
        "assumptions": {
            "trainings_per_year": trainings_per_year,
            "gpu_hourly_cost_usd": gpu_hourly_cost,
            "region_co2_intensity_grams_per_kwh": region_co2_intensity,
            "inferences_per_year": inferences_per_year,
        }
    }


def main():
    """Main execution: load data, estimate costs, save results."""
    # Load carbon tracking data
    carbon_data = load_carbon_tracking()

    print("\n" + "="*60)
    print("ANNUAL CO2 & COST ESTIMATION")
    print("="*60)
    print("\nLoaded carbon tracking data:")
    print("  - Per training: {:.6f} kWh".format(carbon_data['total_energy']))
    print("  - Per training time: {:.2f} hours".format(
        carbon_data['total_time']/3600))

    # Estimate costs
    estimates = estimate_annual_costs(carbon_data)

    # Display results
    print("\n" + "-"*60)
    print("ANNUAL ESTIMATES (52 trainings/year)")
    print("-"*60)

    annual = estimates["annual"]
    print("\nEnergy & Emissions:")
    print("  - Total energy: {:.2f} kWh/year".format(annual['total_energy_kwh']))
    print("  - Total CO2: {:.6f} metric tons/year".format(annual['total_co2_metric_tons']))
    print("  - Total CO2: {:.2f} kg CO2eq/year".format(annual['total_co2_kg']))

    cost = estimates["cost"]
    print("\nCost Breakdown:")
    print("  - Training cost: ${:.2f}/year".format(cost['annual_training_cost_usd']))
    print("  - Infrastructure: ${:.2f}/year".format(cost['annual_infrastructure_overhead_usd']))
    print("  - Inference cost: ${:.2f}/year (1M reqs)".format(cost['annual_inference_cost_usd']))
    print("  - TOTAL: ${:.2f}/year".format(cost['total_annual_cost_usd']))
    print("  - Per training: ${:.2f}".format(cost['cost_per_training_usd']))

    sustainability = estimates["sustainability"]
    print("\nSustainability Impact:")
    print("  - Equivalent to {:.0f} km by car".format(
        sustainability['equivalent_km_car_per_year']))
    print("  - Equivalent to {:.0f} trees needed to offset".format(
        sustainability['equivalent_trees_to_offset_per_year']))
    print("  - CO2 intensity of spending: {:.6f} kg CO2/USD".format(
        sustainability['co2_per_dollar_spent']))
    # Save results
    output_path = Path("results/annual_cost_estimate.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(estimates, f, indent=2)

    print("\n✅ Saved to: {}".format(output_path))
    print("="*60 + "\n")

    return estimates


if __name__ == "__main__":
    main()
