#!/bin/bash
#SBATCH --job-name=cub_hw50
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_head_warmup_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Test fix (3): head warmup. Hyperbolic head trains on contrastive-only
# (geodesic NT-Xent) for the first N epochs before HypHC ramps on.
# Motivation: CIFAR-100 sweep era trained the hyperbolic head during pretrain
# via dual hyperbolic+Euclidean contrastive; current CUB runs skip that, so
# the head starts clustering with a random init — hence the dip.
# ============================================================================

echo "=== CUB-200 supervised + head_warmup_epochs=50 ==="
uv run python train.py --config-name cub200_supervised_baseline \
    clustering.head_warmup_epochs=50 \
    wandb.tags=[cub200,supervised_baseline,head_warmup_test]

module purge
