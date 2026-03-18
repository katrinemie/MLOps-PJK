#!/bin/bash
#SBATCH --job-name=module4
#SBATCH --output=logs/module4_%j.log
#SBATCH --error=logs/module4_%j_err.log
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --time=00:30:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
cd /ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-PJK
mkdir -p logs results

echo "=== 1/3: Quantization benchmark ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" python3 src/quantize_benchmark.py
echo ""

echo "=== 2/3: Batch benchmark ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" python3 src/batch_benchmark.py
echo ""

echo "=== 3/3: Pruning + fine-tuning ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" python3 src/prune_finetune.py
echo ""

echo "=== Done! Results in results/ ==="
ls -la results/
