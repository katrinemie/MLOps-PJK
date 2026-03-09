#!/bin/bash
#SBATCH --job-name=module4_v2
#SBATCH --output=logs/module4_v2_%j.log
#SBATCH --error=logs/module4_v2_%j_err.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --time=00:30:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
WORKDIR="/ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-PJK"

echo "=========================================="
echo "Module 4 v2: Static Quantization + Fixed Pruning"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd "$WORKDIR"
mkdir -p logs results

echo ""
echo ">>> Running STATIC quantization benchmark..."
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/quantize_benchmark.py
RET=$?; [ $RET -ne 0 ] && echo "FAILED quantize with exit code $RET"

echo ""
echo ">>> Running pruning and fine-tuning (fixed)..."
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/prune_finetune.py
RET=$?; [ $RET -ne 0 ] && echo "FAILED pruning with exit code $RET"

echo ""
echo "=========================================="
echo "Module 4 v2 complete!"
echo "=========================================="
