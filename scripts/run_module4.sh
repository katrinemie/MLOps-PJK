#!/bin/bash
#SBATCH --job-name=module4_inference
#SBATCH --output=logs/module4_%j.log
#SBATCH --error=logs/module4_%j_err.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --time=00:30:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
WORKDIR="/ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-PJK"

echo "=========================================="
echo "Module 4: Scalable Inference Experiments"
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd "$WORKDIR"
mkdir -p logs results

# Run quantization benchmark
echo ""
echo ">>> Running quantization benchmark..."
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/quantize_benchmark.py

# Run batch inference benchmark
echo ""
echo ">>> Running batch inference benchmark..."
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/batch_benchmark.py

# Run pruning and fine-tuning
echo ""
echo ">>> Running pruning and fine-tuning..."
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/prune_finetune.py

echo ""
echo "=========================================="
echo "All Module 4 experiments complete!"
echo "Results saved in results/"
echo "=========================================="
