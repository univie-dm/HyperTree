#!/bin/bash
#SBATCH --job-name=cub_unfrz
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_unfreeze_layer4_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Unfreeze layer4 + geodesic NT-Xent on head. Tests whether the collapse is
# due to frozen features making NT-Xent degenerate into pure uniformity
# (positive pair similarity ~constant in W when backbone is already
# augmentation-invariant). With layer4 trainable, NT-Xent has gradient into
# the backbone and becomes a real learning signal again.
# ============================================================================

echo "=== CUB-200 supervised + unfreeze layer4 + geodesic head-only NT-Xent ==="
uv run python train.py --config-name cub200_supervised_baseline \
    architecture.latent_dim=32 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=0.0 \
    clustering.hyperbolic_loss.weight_interval=120 \
    clustering.regularizer.enabled=False \
    clustering.freeze_encoder=True \
    clustering.unfreeze_backbone_from=layer4 \
    clustering.backbone_lr_factor=0.1 \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.euclidean=False \
    clustering.contrastive_loss.weight=1.0 \
    clustering.epochs=300 \
    wandb.tags='[cub200,supervised_baseline,unfreeze_layer4,geodesic_contrastive,latent32]'

module purge
