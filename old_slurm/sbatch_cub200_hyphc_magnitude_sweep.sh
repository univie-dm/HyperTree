#!/bin/bash
#SBATCH --job-name=cub_hyphc_mag
#SBATCH --time=0-06:00:00
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_hyphc_mag_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-11%6

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CUB-200 HypHC magnitude sweep.
# Hypothesis: HypHC end_weight=5.0 is too strong — destroys random head's
# accidental neighborhood-preserving projection before features stabilize.
# Treat HypHC as a gentle secondary signal.
#
# Fixed: layer4 frozen, contrastive=euclidean@1.0, regularizer enabled
# Axis 0: hyphc end_weight      (3) — 0.5, 1.0, 5.0 (current)
# Axis 1: hyphc weight_interval (2) — 120 (fast), 500 (slow warmup)
# Axis 2: regularizer type      (2) — origin, uniformity
# Total: 3 * 2 * 2 = 12
# ============================================================================

HYPHC_WEIGHTS=(0.5 1.0 5.0)
HYPHC_INTERVALS=(120 500)
REG_TYPES=(origin uniformity)

HW_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
HI_IDX=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
RT_IDX=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))

HYPHC_W=${HYPHC_WEIGHTS[$HW_IDX]}
HYPHC_I=${HYPHC_INTERVALS[$HI_IDX]}
REG_TYPE=${REG_TYPES[$RT_IDX]}

echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "hyphc_end_w=${HYPHC_W} | hyphc_interval=${HYPHC_I} | reg_type=${REG_TYPE}"

uv run python train.py --config-name cub200_supervised_baseline \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    clustering.hyperbolic_loss.end_weight=${HYPHC_W} \
    clustering.hyperbolic_loss.weight_interval=${HYPHC_I} \
    clustering.regularizer.enabled=True \
    clustering.regularizer.type=${REG_TYPE} \
    clustering.regularizer.weight=0.05 \
    clustering.unfreeze_backbone_from=null \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.euclidean=True \
    clustering.contrastive_loss.weight=1.0 \
    clustering.epochs=300 \
    wandb.tags='[cub200,supervised_baseline,hyphc_magnitude_sweep]'

module purge
