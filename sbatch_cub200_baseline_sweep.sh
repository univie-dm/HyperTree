#!/bin/bash
#SBATCH --job-name=cub_sup_sweep
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cub_sup_sweep_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-47%8

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CUB-200 supervised-baseline clustering diagnostic sweep (48 runs)
# Goal: find config that preserves the ~42% 1-NN the random hyperbolic head
# already achieves on frozen supervised features, then beats it.
#
# Axis 0: latent_dim          (3) — 32, 128, 512  (CIFAR hinted 32 wins)
# Axis 1: regularizer         (2) — off, on@0.05  (suspect: pulls to origin)
# Axis 2: unfreeze_backbone   (2) — null, layer4  (suspect: degrades features)
# Axis 3: contrastive_weight  (2) — 1.0, 2.0      (needs to dominate reg)
# Axis 4: manifold_precision  (2) — float32, float64  (boundary underflow?)
# Total: 3 * 2 * 2 * 2 * 2 = 48
# ============================================================================

LATENT_DIMS=(32 128 512)
REG_ENABLED=(False True)
UNFREEZES=(null layer4)
C_WEIGHTS=(1.0 2.0)
PRECISIONS=(float32 float64)

LD_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
RG_IDX=$(( (SLURM_ARRAY_TASK_ID / 3) % 2 ))
UF_IDX=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))
CW_IDX=$(( (SLURM_ARRAY_TASK_ID / 12) % 2 ))
PR_IDX=$(( (SLURM_ARRAY_TASK_ID / 24) % 2 ))

L_DIM=${LATENT_DIMS[$LD_IDX]}
REG=${REG_ENABLED[$RG_IDX]}
UNFREEZE=${UNFREEZES[$UF_IDX]}
C_WEIGHT=${C_WEIGHTS[$CW_IDX]}
PRECISION=${PRECISIONS[$PR_IDX]}

echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "latent_dim=${L_DIM} | reg=${REG} | unfreeze=${UNFREEZE} | c_weight=${C_WEIGHT} | manifold_prec=${PRECISION}"

uv run python train.py --config-name cub200_supervised_baseline \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    architecture.latent_dim=${L_DIM} \
    architecture.manifold_precision=${PRECISION} \
    clustering.regularizer.enabled=${REG} \
    clustering.unfreeze_backbone_from=${UNFREEZE} \
    clustering.contrastive_loss.weight=${C_WEIGHT} \
    clustering.epochs=400 \
    clustering.hyperbolic_loss.weight_interval=150 \
    wandb.tags='[cub200,supervised_baseline,diagnostic_sweep]'

module purge
