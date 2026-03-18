#!/bin/bash
#SBATCH --job-name=gen_figures
#SBATCH --output=logs/figures_%j.log
#SBATCH --error=logs/figures_%j_err.log
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
cd /ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-PJK
mkdir -p logs

echo "Generating figures..."
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/generate_figures.py
echo "Done. Exit code: $?"
