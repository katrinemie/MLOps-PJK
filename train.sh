#!/bin/bash
#SBATCH --job-name=train-catdog
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=12:00:00

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p logs models

# Pull data fra DVC
singularity exec --nv singularity/pytorch.sif \
    dvc pull

# Kør træning
singularity exec --nv singularity/pytorch.sif \
    bash -c "PYTHONPATH='$ROOT/src' python -m train --config configs/config.yaml $*"
