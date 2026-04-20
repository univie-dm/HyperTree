#!/bin/bash
#SBATCH --job-name=cub_eucl
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_head_only_euclidean_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Euclidean head control: same pipeline as 524335 / 524336 (head-only
# contrastive, frozen backbone, latent=32) but manifold=Euclidean.
# Distinguishes "hyperbolic geometry is destructive" from "training any
# projection on near-JL-preserving frozen features drifts away from optimum".
#
# If this ALSO collapses -> pipeline-level issue (NT-Xent + frozen features)
# If this PRESERVES / IMPROVES -> hyperbolic-specific failure mode
# ============================================================================

echo "=== CUB-200 supervised + head-only contrastive, Euclidean, latent=32 ==="
uv run python train.py --config-name cub200_supervised_baseline \
    architecture.manifold_name=Euclidean \
    architecture.latent_dim=32 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=0.0 \
    clustering.hyperbolic_loss.weight_interval=120 \
    clustering.regularizer.enabled=False \
    clustering.unfreeze_backbone_from=null \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.euclidean=False \
    clustering.contrastive_loss.weight=1.0 \
    clustering.epochs=300 \
    wandb.tags='[cub200,supervised_baseline,head_only_contrastive,euclidean,latent32]'

module purge
