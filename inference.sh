#!/bin/bash
#SBATCH --job-name=infer-catdog
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CONTAINER=/ceph/container/pytorch/pytorch_26.01.sif

mkdir -p logs

srun singularity exec --nv "$CONTAINER" \
    bash -c "PYTHONPATH='$ROOT/src' python -m inference --model models/best_model.pt $*"
