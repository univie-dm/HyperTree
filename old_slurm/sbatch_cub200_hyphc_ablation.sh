#!/bin/bash
#SBATCH --job-name=cub_noHC
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_hyphc_ablation_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# HypHC ablation: run the winning recipe with hyperbolic_loss.end_weight=0.
# Winners all lose 0.42 -> 0.20 — is HypHC destroying the NN structure that
# supervised features already carry? If R@1 stays near 0.42 here, HypHC is
# a net negative on CUB under the supervised baseline.
#
# Winning recipe pinned:
#   regularizer.type=uniformity, regularizer.weight=0.05
#   unfreeze_backbone_from=null (fully frozen)
#   contrastive_loss.enabled=True, contrastive_loss.euclidean=True
# Ablation: hyperbolic_loss.end_weight=0 (HypHC effectively off)
# ============================================================================

echo "=== CUB-200 supervised + HypHC ablation (end_weight=0) ==="
uv run python train.py --config-name cub200_supervised_baseline \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=0.0 \
    clustering.hyperbolic_loss.weight_interval=120 \
    clustering.regularizer.type=uniformity \
    clustering.regularizer.weight=0.05 \
    clustering.unfreeze_backbone_from=null \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.euclidean=True \
    clustering.epochs=300 \
    wandb.tags='[cub200,supervised_baseline,hyphc_ablation]'

module purge
