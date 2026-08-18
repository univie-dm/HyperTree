#!/bin/bash
#SBATCH --job-name=c100_dim_abl
#SBATCH --time=10:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/c100_dim_abl_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel,dgx-h100-em2,vader
#SBATCH --array=0-11%4

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CIFAR-100 LOSS ABLATION ACROSS EMBEDDING DIMENSIONALITIES (d=32, d=128).
#
# Answers "what about other embedding dimensionalities?" on the AC-1 loss
# decomposition. The triad currently exists at CIFAR-100 d=16, TinyImageNet
# d=16, and ImageNet-1k d=128. This adds CIFAR-100 d=32 and d=128, where the
# joint arms already exist (baselines/rebuild/c100_poin_d{32,128}_*).
#
# Recipe matches sbatch_rebuild_c100_poin_d{32,128}.sh exactly except for the
# loss weights, so the three arms share one encoder and one anchor per dim.
#
#   conly : L_H end_weight = 0,  contrastive on at 0.1
#   lhonly: L_H 1.0 -> 5.0,      contrastive off
#
# NOTE on lhonly start_weight: it must be > 0. With contrastive disabled and
# start_weight=0 the total loss is identically zero at epoch 0 and backward
# fails with "element 0 of tensors does not require grad". This cost two failed
# ImageNet attempts; do not set it to 0 for any contrastive-off arm.
#
# All six SimCLR encoders are already cached (the cache hash includes
# latent_dim), so these are clustering-only, ~2.5 h each.
#
# Array decoding: task = dim_idx*6 + arm_idx*3 + seed_idx
# ============================================================================

DIMS=(32 128)
ARMS=(conly lhonly)
SEEDS=(1 2 3)

DIM_IDX=$((SLURM_ARRAY_TASK_ID / 6))
REST=$((SLURM_ARRAY_TASK_ID % 6))
ARM_IDX=$((REST / 3))
SEED_IDX=$((REST % 3))

DIM=${DIMS[$DIM_IDX]}
ARM=${ARMS[$ARM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [ "$ARM" = "conly" ]; then
  HL_START=0.0; HL_END=0.0; C_EN=True;  C_W=0.1
else
  HL_START=1.0; HL_END=5.0; C_EN=False; C_W=0.0
fi

export HYP_DUMP_EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/rebuild/c100_${ARM}_d${DIM}_seed${SEED}
echo "=== c100_dim_abl task ${SLURM_ARRAY_TASK_ID}: d=${DIM} arm=${ARM} seed=${SEED} ==="

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    experiment.seed=${SEED} \
    experiment.batch_size=1024 \
    experiment.log_interval=10 \
    experiment.gradient_clip_value=5 \
    experiment.optimizer=Adam \
    architecture.model_type=resnet50 \
    architecture.latent_dim=${DIM} \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    architecture.projection_layer_dims=[512,256,128] \
    architecture.hyperbolic_smoothing=50.0 \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.learning_rate=0.003 \
    pretraining.contrastive_loss.enabled=True \
    pretraining.contrastive_loss.weight=1.0 \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=300 \
    clustering.learning_rate=0.0001 \
    clustering.freeze_encoder=True \
    clustering.contrastive_loss.enabled=${C_EN} \
    clustering.contrastive_loss.weight=${C_W} \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=${HL_START} \
    clustering.hyperbolic_loss.end_weight=${HL_END} \
    clustering.hyperbolic_loss.weight_interval=250 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.temperature_interval=250 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[cifar100,poincare_d'${DIM}','${ARM}',dim_loss_ablation,seed'${SEED}']'

module purge
