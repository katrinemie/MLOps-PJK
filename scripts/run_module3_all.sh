#!/bin/bash
#SBATCH --job-name=module3
#SBATCH --output=logs/module3_%j.log
#SBATCH --error=logs/module3_%j_err.log
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --cpus-per-task=15
#SBATCH --time=02:00:00

CONTAINER="/ceph/container/pytorch/pytorch_26.02.sif"
PROJECT="/ceph/home/student.aau.dk/gd65nz/MLOPS_DAKI4/MLOps-PJK"
cd "$PROJECT"
mkdir -p logs results

# Kort træning for benchmark: 3 epochs
export EPOCHS=3

echo "=============================================="
echo "MODULE 3: Scalable Training Benchmarks"
echo "=============================================="
echo "GPUs: $(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# --- 1. Single GPU, NO AMP (baseline) ---
echo "=== 1/5: Single GPU, NO AMP (baseline) ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/train_ddp_benchmark.py --config configs/config.yaml --no-amp --gpus 1 --epochs $EPOCHS
echo ""

# --- 2. Single GPU, WITH AMP ---
echo "=== 2/5: Single GPU, WITH AMP ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/train_ddp_benchmark.py --config configs/config.yaml --gpus 1 --epochs $EPOCHS
echo ""

# --- 3. Multi-GPU DDP, NO AMP ---
echo "=== 3/5: 2 GPUs DDP, NO AMP ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    torchrun --standalone --nproc_per_node=2 \
    src/train_ddp_benchmark.py --config configs/config.yaml --no-amp --gpus 2 --epochs $EPOCHS
echo ""

# --- 4. Multi-GPU DDP, WITH AMP ---
echo "=== 4/5: 2 GPUs DDP, WITH AMP ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    torchrun --standalone --nproc_per_node=2 \
    src/train_ddp_benchmark.py --config configs/config.yaml --gpus 2 --epochs $EPOCHS
echo ""

# --- 5. Single GPU, AMP + sammenligning ---
echo "=== 5/5: Sammenfatning ==="
singularity exec --nv --env PYTHONNOUSERSITE=1 "$CONTAINER" \
    python3 src/summarize_module3.py
echo ""

echo "=== DONE ==="
