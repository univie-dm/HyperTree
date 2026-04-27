#!/bin/bash
#SBATCH --job-name=c100_d16_hist_revert
#SBATCH --time=16:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_d16_hist_revert_%A_%a.log
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
# CIFAR-100 d=16 Poincaré: HISTORICAL CONFIG REVERT TEST.
#
# Hypothesis: the spike-and-crash trajectory in worldly-oath-1224 (peak hDP
# 0.89 at ce=11, crash to 0.73 by ce=300) is caused by config drift since
# the *-16-Ours runs (ulzojs72 etc., April 5) which finished at 0.901 with
# no crash.
#
# Wandb config diff (ulzojs72 vs worldly-oath-1224) flagged three drifts:
#   1. sampling_eps: 0.01 -> 0.001 (10x smaller)
#   2. use_backbone_knn_similarities: absent (False) -> True
#   3. freeze_encoder: True (default per LESSONS line 109) -> False
#
# This sweep reverts all three and runs the schedule the historical recipe
# used (end_weight=5, weight_interval=250). 3 seeds × 1 condition.
#
# Decision rule:
#   * Final hDP_ward ~0.90 with no crash -> config drift confirmed; restore
#     to paper_main and stop debugging late-trajectory code.
#   * Crash still happens -> suspect list narrows to genuine code drift in
#     the post-ce=11 trajectory; need git bisect.
#
# Reuses cached SimCLR encoder (overwrite=False) — same hash as worldly-oath
# and the schedfix runs.
# ============================================================================

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== c100_d16_hist_revert task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

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
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.weight_interval=250 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.temperature_interval=250 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,poincare_d16_hist_revert,seed'${SEED}']'

module purge
