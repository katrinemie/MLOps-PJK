#!/bin/bash
#SBATCH --job-name=quant_v3
#SBATCH --output=logs/quant_v3_%j.log
#SBATCH --error=logs/quant_v3_%j_err.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --time=00:15:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
cd /ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-PJK
mkdir -p logs results

echo "Running FX quantization benchmark..."
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/quantize_benchmark.py
