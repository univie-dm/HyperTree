#!/bin/bash
#SBATCH --job-name=c100_wideb
#SBATCH --time=16:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/c100_wideb_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Keep the embedding off the ball boundary. With the float64 default (1e-8)
# the trained model saturates at 1-||x|| ~ 1e-8, where the geodesic distance
# formula loses ~8 digits and 98-99% of pairwise distances collapse into ties.
# 1e-4 keeps (1-c||x||^2) at ~2e-4, fully resolved in float64.
export HYP_MAX_ENORM_EPS=1e-4

# CIFAR-100 d=16 hist_revert run for embedding dump (LCA wrong-rate analysis).
# 50 ep clustering instead of 300 (trajectory plateaus by ~ce=20 under
# hist_revert). Reuses cached SimCLR encoder (overwrite=False) — same hash
# as the 3-seed hist_revert sweep. 3 seeds × 1 dataset = 3 runs.
#
# Dump goes to baselines/lca_wrong_rate/c100_d16_wideb_seed${SEED}_emb.pt via the
# HYP_DUMP_EMB env var (handled in src/models/abstract.py).

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DUMP_DIR=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/c100_d16_wideboundary
mkdir -p ${DUMP_DIR}
# HYP_DUMP_EMB is a prefix: per-eval saves to ${prefix}_ce${ep}.pt
export HYP_DUMP_EMB=${DUMP_DIR}/c100_d16_wideb_seed${SEED}

echo "=== lca_dump_c100 task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} dump=${HYP_DUMP_EMB} ==="

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
    experiment.log_interval=5 \
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
    wandb.tags='[cifar100,poincare_d16_wide_boundary,seed'${SEED}']'

module purge
