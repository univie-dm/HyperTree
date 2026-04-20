#!/bin/bash
#SBATCH --job-name=c100_sweep
#SBATCH --time=2-00:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_sweep_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-31%8

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CIFAR-100 SimCLR sweep — 32 configs
# Key axes informed by dev-branch comparison:
#   freeze_encoder was the dominant factor for hierarchical DP
# ============================================================================
# Axis 0: freeze_encoder (2)              - False (old best), True
# Axis 1: use_backbone_knn_similarities (2) - False (old), True (new)
# Axis 2: pretrain contrastive temp (2)   - 0.1 (current), 0.5 (old best)
# Axis 3: latent_dim (2)                  - 32 (old best), 64
# Axis 4: pretrain augmentation (2)       - null (old best), byol
# Total: 2^5 = 32

FREEZE_ENCODERS=(False True)
BACKBONE_KNNS=(False True)
PRETRAIN_TEMPS=(0.1 0.5)
LATENT_DIMS=(32 64)
AUGMENTATIONS=(null byol)

# Decode array index into grid coordinates
FE_IDX=$((SLURM_ARRAY_TASK_ID % 2))
BK_IDX=$(( (SLURM_ARRAY_TASK_ID / 2) % 2 ))
PT_IDX=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
LD_IDX=$(( (SLURM_ARRAY_TASK_ID / 8) % 2 ))
AU_IDX=$(( (SLURM_ARRAY_TASK_ID / 16) % 2 ))

FREEZE=${FREEZE_ENCODERS[$FE_IDX]}
BK_KNN=${BACKBONE_KNNS[$BK_IDX]}
P_TEMP=${PRETRAIN_TEMPS[$PT_IDX]}
L_DIM=${LATENT_DIMS[$LD_IDX]}
AUG=${AUGMENTATIONS[$AU_IDX]}

echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "freeze_encoder=${FREEZE} | backbone_knn=${BK_KNN} | pretrain_temp=${P_TEMP} | latent_dim=${L_DIM} | augmentation=${AUG}"

python train.py +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    experiment.seed=42 \
    experiment.batch_size=1024 \
    experiment.log_interval=10 \
    architecture.latent_dim=${L_DIM} \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=1200 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=${AUG} \
    pretraining.contrastive_loss.temperature=${P_TEMP} \
    pretraining.accumulate_gradients=2 \
    clustering.epochs=1000 \
    clustering.freeze_encoder=${FREEZE} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=${BK_KNN} \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[cifar100,sweep]'

module purge
