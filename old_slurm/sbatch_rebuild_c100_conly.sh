#!/bin/bash
#SBATCH --job-name=rb_c100_conly
#SBATCH --time=16:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/rb_c100_conly_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel,dgx-h100-em2
#SBATCH --array=0-2%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CIFAR-100 d=16 Poincare: CONTRASTIVE-ONLY ablation arm (rebuttal AC-1,
# Gy9D-W2/Q3, Y8o7-Q2).
#
# Third arm of the loss-isolation triad. Identical to the joint headline
# recipe and to the L_H-only arm except for the loss weights:
#   joint   : L_C w=0.1  +  L_H w=1->5     -> hDP_ward 0.550 +- 0.111
#   L_H-only: L_C off    +  L_H w=1->5     -> hDP_ward 0.331 (R@1 collapses)
#   THIS    : L_C w=0.1  +  L_H w=0        -> ?
#
# Purpose: the ce=0 anchor (0.299) has a RANDOMLY INITIALIZED head, so the
# joint delta of +0.25 cannot attribute between the two losses. This arm
# trains the same head with contrastive supervision only, isolating how much
# of the gain L_C accounts for on its own.
#
# w = 1/(1+d) throughout (use_backbone_knn_similarities=False), matching the
# other two arms so the triad is comparable.
#
# Reuses the cached SimCLR encoder (overwrite=False) - same hash as the
# joint and L_H-only arms.
# ============================================================================

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

export HYP_DUMP_EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/rebuild/c100_conly_d16_seed${SEED}
echo "=== rb_c100_conly task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    experiment.seed=${SEED} \
    experiment.batch_size=1024 \
    experiment.log_interval=10 \
    experiment.gradient_clip_value=5 \
    experiment.optimizer=Adam \
    architecture.model_type=resnet50 \
    architecture.latent_dim=16 \
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
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=0.0 \
    clustering.hyperbolic_loss.weight_interval=0 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.temperature_interval=250 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,poincare_d16,conly,ablation,seed'${SEED}']'

module purge
