#!/bin/bash
#SBATCH --job-name=cub_reg_sweep
#SBATCH --time=0-08:00:00
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_reg_sweep_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-23%8

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CUB-200 regularizer sweep — Yang et al. 2024 outer-shell isotropy vs origin
# attraction vs off, crossed with unfreeze strategy and contrastive setting.
#
# Axis 0: regularizer type    (3) — off, origin, uniformity
# Axis 1: regularizer weight  (2) — 0.05, 0.5
# Axis 2: unfreeze_backbone   (2) — null (frozen), layer4
# Axis 3: contrastive         (2) — off, euclidean (weight 1.0)
# Total: 3 * 2 * 2 * 2 = 24
#
# Expect uniformity to preserve 1-NN better than origin under layer4 unfreeze
# (allows radial hierarchy, prevents angular collapse).
# ============================================================================

REG_TYPES=(off origin uniformity)
REG_WEIGHTS=(0.05 0.5)
UNFREEZES=(null layer4)
CONTRASTIVES=(off euclidean)

RT_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
RW_IDX=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
UF_IDX=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))
CO_IDX=$(( (SLURM_ARRAY_TASK_ID / 12) % 2 ))

REG_TYPE=${REG_TYPES[$RT_IDX]}
REG_WEIGHT=${REG_WEIGHTS[$RW_IDX]}
UNFREEZE=${UNFREEZES[$UF_IDX]}
CONTRASTIVE=${CONTRASTIVES[$CO_IDX]}

# Translate reg_type=off to enabled=False; any type with enabled=False is inert
if [ "$REG_TYPE" = "off" ]; then
    REG_ENABLED=False
    REG_TYPE_PASS=origin  # placeholder, won't run
else
    REG_ENABLED=True
    REG_TYPE_PASS=$REG_TYPE
fi

if [ "$CONTRASTIVE" = "off" ]; then
    C_ENABLED=False
    C_EUCL=False
else
    C_ENABLED=True
    C_EUCL=True
fi

echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "reg_type=${REG_TYPE} | reg_w=${REG_WEIGHT} | unfreeze=${UNFREEZE} | contrastive=${CONTRASTIVE}"

uv run python train.py --config-name cub200_supervised_baseline \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    clustering.regularizer.enabled=${REG_ENABLED} \
    clustering.regularizer.type=${REG_TYPE_PASS} \
    clustering.regularizer.weight=${REG_WEIGHT} \
    clustering.unfreeze_backbone_from=${UNFREEZE} \
    clustering.contrastive_loss.enabled=${C_ENABLED} \
    clustering.contrastive_loss.euclidean=${C_EUCL} \
    clustering.epochs=300 \
    clustering.hyperbolic_loss.weight_interval=120 \
    wandb.tags='[cub200,supervised_baseline,regularizer_sweep,uniformity]'

module purge
