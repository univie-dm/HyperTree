#!/bin/bash
#SBATCH --job-name=c100_poinc_d16
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_poinc_d16_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-2%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CIFAR-100 PoincareBall d=16 — reruns the *-16-Ours recipe on current HEAD so
# the paper main table has Poincaré d=16 numbers that match the eval pipeline
# used for the Euclidean d=16 baseline (533763) and the SeCu/SCAN baselines.
# The historical *-16-Ours runs (ulzojs72 et al., April 5) sit at hDP_ward ≈
# 0.901 / hDP_complete ≈ 0.909 but were computed on dev tip c0b4a5e with older
# hDP machinery — not apples-to-apples with the current eval.
#
# Recipe (identical to sbatch_cifar100_euclidean_d16.sh except manifold + HypHC):
#   SimCLR 800ep pretrain (lr=3e-3) + clustering 300ep (lr=1e-4)
#   latent_dim=16, projection head [512,256,128]
#   Contrastive enabled DURING clustering (weight=0.1, t=0.5) — no freeze
#   HypHC ramped sw=1.0 -> ew=5.0, temp 0.2 -> 0.1 (defaults in config.yaml)
#   Seeds 1, 2, 3 to match the euclidean_d16_ours runs
# ============================================================================

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== c100_poinc_d16 task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

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
    clustering.freeze_encoder=False \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    architecture.hyperbolic_smoothing=50.0 \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,poincare_d16_ours,paper_main,seed'${SEED}']'

module purge
