#!/bin/bash
#SBATCH --job-name=tinyimg_sweep
#SBATCH --time=2-00:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tiny_imagenet_sweep_%A_%a.log
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
# Tiny ImageNet SimCLR sweep — 32 configs
# 200 classes, 64x64 — bigger than CIFAR but still small-image regime
# ============================================================================
# Axis 0: model_type (2)                  - resnet18, resnet50
# Axis 1: latent_dim (2)                  - 64, 128
# Axis 2: freeze_encoder (2)              - False, True
# Axis 3: use_backbone_knn_similarities (2) - False, True
# Axis 4: pretrain contrastive temp (2)   - 0.1, 0.5
# Total: 2^5 = 32

MODELS=(resnet18 resnet50)
LATENT_DIMS=(64 128)
FREEZE_ENCODERS=(False True)
BACKBONE_KNNS=(False True)
PRETRAIN_TEMPS=(0.1 0.5)

# Decode array index into grid coordinates
MD_IDX=$((SLURM_ARRAY_TASK_ID % 2))
LD_IDX=$(( (SLURM_ARRAY_TASK_ID / 2) % 2 ))
FE_IDX=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
BK_IDX=$(( (SLURM_ARRAY_TASK_ID / 8) % 2 ))
PT_IDX=$(( (SLURM_ARRAY_TASK_ID / 16) % 2 ))

MODEL=${MODELS[$MD_IDX]}
L_DIM=${LATENT_DIMS[$LD_IDX]}
FREEZE=${FREEZE_ENCODERS[$FE_IDX]}
BK_KNN=${BACKBONE_KNNS[$BK_IDX]}
P_TEMP=${PRETRAIN_TEMPS[$PT_IDX]}

echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "model=${MODEL} | latent_dim=${L_DIM} | freeze_encoder=${FREEZE} | backbone_knn=${BK_KNN} | pretrain_temp=${P_TEMP}"

python train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=42 \
    experiment.batch_size=512 \
    experiment.log_interval=10 \
    architecture.model_type=${MODEL} \
    architecture.latent_dim=${L_DIM} \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.contrastive_loss.temperature=${P_TEMP} \
    clustering.epochs=600 \
    clustering.freeze_encoder=${FREEZE} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=${BK_KNN} \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,sweep]'

module purge
