#!/bin/bash
#SBATCH --job-name=test-catdog
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=01:00:00

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

mkdir -p logs

singularity exec --nv singularity/pytorch.sif \
    bash -c "PYTHONPATH='$ROOT/src' python -m evaluate --config configs/config.yaml --model models/best_model.pt $*"
