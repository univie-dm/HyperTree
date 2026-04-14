#!/bin/bash
#SBATCH --job-name=cub_sup
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_supervised_baseline_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CUB-200 clustering with supervised ImageNet weights (1 GPU).
# Tests clustering collapse fixes (start_weight=0, origin regularizer)
# independently from the self-supervised pretrain.
# ============================================================================

echo "=== CUB-200 supervised baseline with clustering fixes ==="
uv run python train.py --config-name cub200_supervised_baseline

module purge
