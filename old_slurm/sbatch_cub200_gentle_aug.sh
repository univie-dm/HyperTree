#!/bin/bash
#SBATCH --job-name=cub_gtlA
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_gentle_aug_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Same head-only geodesic NT-Xent setup as 524335 (Poincaré, latent=32, frozen
# backbone, no regularizer, HypHC off) but with byol_scale=[0.6, 1.0] instead
# of the default [0.2, 1.0]. Tests whether aggressive RandomResizedCrop on
# CUB (where birds are often a small portion of the image) is producing
# degenerate positive pairs that poison NT-Xent.
# ============================================================================

echo "=== CUB-200 head-only geodesic NT-Xent + gentler BYOL crop (scale=[0.6,1.0]) ==="
uv run python train.py --config-name cub200_supervised_baseline \
    architecture.latent_dim=32 \
    pretraining.byol_scale=[0.6,1.0] \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=0.0 \
    clustering.hyperbolic_loss.weight_interval=120 \
    clustering.regularizer.enabled=False \
    clustering.unfreeze_backbone_from=null \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.euclidean=False \
    clustering.contrastive_loss.weight=1.0 \
    clustering.epochs=300 \
    wandb.tags='[cub200,supervised_baseline,head_only_contrastive,latent32,gentle_aug]'

module purge
