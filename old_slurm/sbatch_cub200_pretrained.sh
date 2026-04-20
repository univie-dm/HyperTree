#!/bin/bash
#SBATCH --job-name=cub200_pretrained
#SBATCH --time=1-00:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub200_pretrained_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CUB-200 with ImageNet-pretrained ResNet50 backbone
# Encoder frozen, only projection head + clustering trained
# 400 pretrain epochs (just projection head) + 600 clustering epochs
# ============================================================================

echo "=== CUB-200 Pretrained ResNet50 ==="

python train.py --config-name cub200_pretrained

module purge
