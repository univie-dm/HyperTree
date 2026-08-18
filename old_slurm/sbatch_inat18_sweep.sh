#!/bin/bash
#SBATCH --job-name=inat_sweep
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/inat18_sweep_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-15%8

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 supervised-baseline sweep — 16 configs
# 437k images / 8142 species with real taxonomic hierarchy.
# Fast-filter setup: 1 pretrain epoch (shared via hash), 8 clustering epochs.
# Runs don't recover from collapse, so short budgets are the right tool.
#
# Axes (2^4 = 16):
#   Axis 0: unfreeze        — null (fully frozen) vs layer4
#   Axis 1: euclidean_contrast — True (pre-hyperbolic NT-Xent) vs False (geodesic)
#   Axis 2: hyphc_on        — end_weight 0.0 (HypHC off) vs 5.0 (HypHC on)
#   Axis 3: regularizer     — off vs uniformity@0.05
# ============================================================================

UNFREEZES=(null layer4)
EUCLIDEANS=(True False)
HYPHC_WEIGHTS=(0.0 5.0)
REG_CONFIGS=(off uniformity)

UN_IDX=$((SLURM_ARRAY_TASK_ID % 2))
EU_IDX=$(( (SLURM_ARRAY_TASK_ID / 2) % 2 ))
HC_IDX=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
RG_IDX=$(( (SLURM_ARRAY_TASK_ID / 8) % 2 ))

UNFREEZE=${UNFREEZES[$UN_IDX]}
EUCLIDEAN=${EUCLIDEANS[$EU_IDX]}
HC_W=${HYPHC_WEIGHTS[$HC_IDX]}
REG=${REG_CONFIGS[$RG_IDX]}

if [[ "${REG}" == "off" ]]; then
    REG_ENABLED=False
    REG_TYPE=origin   # unused, but must be valid
    REG_WEIGHT=0.0
else
    REG_ENABLED=True
    REG_TYPE=uniformity
    REG_WEIGHT=0.05
fi

echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "unfreeze=${UNFREEZE} | euclidean=${EUCLIDEAN} | hyphc_w=${HC_W} | reg=${REG}"

uv run python train.py --config-name inat18_supervised_baseline \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    pretraining.epochs=1 \
    clustering.epochs=8 \
    clustering.unfreeze_backbone_from=${UNFREEZE} \
    clustering.contrastive_loss.euclidean=${EUCLIDEAN} \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.weight_interval=4 \
    clustering.hyperbolic_loss.temperature_interval=4 \
    clustering.lr_scheduler.interval=2 \
    clustering.regularizer.enabled=${REG_ENABLED} \
    clustering.regularizer.type=${REG_TYPE} \
    clustering.regularizer.weight=${REG_WEIGHT} \
    wandb.tags='[inat18,supervised_baseline,sweep_v1]'

module purge
