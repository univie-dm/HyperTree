#!/bin/bash
#SBATCH --job-name=ti_d16_wideb
#SBATCH --time=1-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/ti_d16_wideb_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-2%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Wide-boundary rerun: TinyImageNet d=16 is the most boundary-degenerate
# checkpoint we have (0.12% distinct pairwise distances). 1e-4 keeps the
# embedding numerically resolved so the complete-linkage cell -- which
# reads 0.740 on the saturated path and 0.500 on the extended-precision
# path -- can be settled.
export HYP_MAX_ENORM_EPS=1e-4


export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# TinyImageNet H2H at latent_dim=16 — fills the dimensionality-claim gap.
# Paper claims "exceptionally low-dimensional"; CIFAR-100 has d=16/32/128
# coverage, but TI is d=128 only. This run gives the matching low-d row.
#
# Mirrors sbatch_tiny_imagenet_h2h_warm.sh except --architecture.latent_dim=16.
# Pretrain hash differs from d=128 cache, so SimCLR will retrain (800 ep).
# 3 conditions × 3 seeds = 9 runs.
# ============================================================================

SEEDS=(42 43 44)
SEED_IDX=$SLURM_ARRAY_TASK_ID
COND_IDX=0
SEED=${SEEDS[$SEED_IDX]}

case $COND_IDX in
  0) MANIFOLD=PoincareBall; HC_W=5.0; CND_TAG=poincare_hyphc      ;;
  1) MANIFOLD=PoincareBall; HC_W=0.0; CND_TAG=poincare_no_hyphc   ;;
  2) MANIFOLD=Euclidean;    HC_W=0.0; CND_TAG=euclidean_baseline  ;;
esac

mkdir -p /srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/ti_d16_wideboundary
export HYP_DUMP_EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/ti_d16_wideboundary/ti_d16_wideb_seed${SEED}
echo "=== rb_ti_d16 task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ${CND_TAG} d=16 ==="

uv run python train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=512 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=16 \
    architecture.manifold_name=${MANIFOLD} \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=600 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,h2h_warm,d16,wide_boundary,'${CND_TAG}',seed'${SEED}']'

module purge
