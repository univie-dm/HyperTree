#!/bin/bash
#SBATCH --job-name=c100_root_false
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_d16_root_false_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel,dgx-h100-em2
#SBATCH --array=0

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CIFAR-100 d=16 hist_revert — test run with dendrogrampurity_use_root=False.
#
# Goal: see how much the CIFAR-100 hDP_ward absolute shifts when we exclude
# the root level from the hierarchical-DP computation (the convention used
# on TI / ImageNet / iNat). The 3-seed hist_revert sweep landed at
# 0.846 ± 0.029 with use_root=True; this single-seed probe quantifies the
# delta from flipping the flag, before we commit to a full 5-seed re-eval
# of CIFAR-100.
#
# Single seed (1) only. 50 clustering epochs (trajectory plateaus by ~ce=20
# under hist_revert; 50 is more than enough to read the final-epoch value).
# SimCLR encoder is cached (overwrite=False) — same hash as the main
# hist_revert sweep — so this run skips the 800 pretrain epochs.
#
# Decision rule:
#   * |Δ| ≥ 0.03 with stable trajectory  → consider flipping the flag for
#     the paper main table (need to re-eval HIER baselines under the same
#     flag for fair comparison).
#   * |Δ| < 0.03                          → keep use_root=True; document
#     the small delta as a footnote.
# ============================================================================

SEED=1

echo "=== c100_d16_root_false test: seed=${SEED}, use_root=False, 50 epochs ==="

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
    clustering.epochs=50 \
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
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[cifar100,poincare_d16_hist_revert,root_false_test,seed'${SEED}']'

module purge
