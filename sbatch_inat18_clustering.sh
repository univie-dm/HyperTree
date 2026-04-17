#!/bin/bash
#SBATCH --job-name=inat_clust
#SBATCH --time=1-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/inat18_clustering_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 clustering run on supervised ImageNet ResNet50. Skips contrastive
# pretraining — backbone adaptation happens in the clustering stage via its
# own contrastive loss (contrastive_loss.enabled=True) alongside the hyperbolic
# clustering loss. unfreeze_backbone_from=layer4 keeps most of R50 frozen.
# ============================================================================

echo "=== iNat18 clustering on supervised R50 (no separate pretrain) ==="
uv run python train.py --config-name inat18_supervised_baseline \
    pretraining.epochs=0 \
    wandb.tags='[inat18,clustering,supervised_r50]'

module purge
