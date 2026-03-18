#!/bin/bash
# Multi-node DDP training launch script for DAKI kluster.
#
# Run this script on each node with the correct NODE_RANK:
#   Node 0 (master): NODE_RANK=0 bash scripts/launch_multinode.sh
#   Node 1:          NODE_RANK=1 bash scripts/launch_multinode.sh
#   Node 2:          NODE_RANK=2 bash scripts/launch_multinode.sh
#
# Environment variables (set before running):
#   MASTER_ADDR  - IP/hostname of node 0 (default: daki-master)
#   MASTER_PORT  - Free port on master (default: 29500)
#   NODE_RANK    - This node's rank: 0, 1, or 2 (required)
#   NNODES       - Total number of nodes (default: 3)
#   NPROC        - GPUs per node (default: 1)

MASTER_ADDR=${MASTER_ADDR:-daki-master}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-3}
NPROC=${NPROC:-1}

if [ -z "$NODE_RANK" ]; then
    echo "ERROR: NODE_RANK must be set (0, 1, or 2)"
    exit 1
fi

echo "=== Multi-node DDP Training ==="
echo "  Master:    $MASTER_ADDR:$MASTER_PORT"
echo "  Nodes:     $NNODES"
echo "  GPUs/node: $NPROC"
echo "  Node rank: $NODE_RANK"
echo "==============================="

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$NODE_RANK \
    src/train_ddp.py --config configs/config.yaml
