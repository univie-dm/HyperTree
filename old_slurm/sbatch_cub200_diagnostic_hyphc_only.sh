#!/bin/bash
#SBATCH --job-name=cub_diag
#SBATCH --time=0-08:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_diagnostic_hyphc_only_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Diagnostic: can HypHC loss alone preserve the 42% baseline?
# - Backbone fully frozen (no layer4 unfreeze) -> isolates clustering head
# - Contrastive disabled -> removes instance-discrimination drift
# - HypHC ramps from 0, regularizer on
# Expectation: if HypHC is benign, 1-NN stays near 42%. If it drops anyway,
# the problem is in the hyperbolic clustering loss itself.
# ============================================================================

uv run python train.py --config-name cub200_supervised_baseline \
  clustering.unfreeze_backbone_from=null \
  clustering.contrastive_loss.enabled=False \
  clustering.contrastive_loss.weight=0.0 \
  wandb.tags='[cub200,diagnostic,hyphc_only,frozen_backbone]'

module purge
