#!/bin/bash
#SBATCH --job-name=c100_h2h
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_h2h_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-8%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CIFAR-100 h2h sweep mirroring sbatch_tiny_imagenet_h2h_warm.sh so we have
# matching 3-seed manifold comparisons on both datasets. CIFAR-100 already
# hits hDP_ward >= 0.77 with the *cold* HypHC schedule (st=0.2, et=0.1),
# which differs from the TI optimum (st=1.0, et=0.5). Keep the cold schedule
# here -- this is the CIFAR-100 h2h, not a re-tuning sweep.
#
# 3 conditions x 3 seeds = 9 runs.
# task_idx = seed_idx*3 + cond_idx
#   seed_idx 0,1,2 = seeds 42, 43, 44
#   cond_idx 0 = PoincareBall + HypHC(ew=5)  poincare_hyphc
#   cond_idx 1 = PoincareBall + HypHC(ew=0)  poincare_no_hyphc
#   cond_idx 2 = Euclidean    + HypHC(ew=0)  euclidean_baseline
# ============================================================================

SEEDS=(42 43 44)
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_IDX]}

case $COND_IDX in
  0) MANIFOLD=PoincareBall; HC_W=5.0; CND_TAG=poincare_hyphc      ;;
  1) MANIFOLD=PoincareBall; HC_W=0.0; CND_TAG=poincare_no_hyphc   ;;
  2) MANIFOLD=Euclidean;    HC_W=0.0; CND_TAG=euclidean_baseline  ;;
esac

echo "=== c100_h2h task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ${CND_TAG} ==="

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    experiment.seed=${SEED} \
    experiment.batch_size=512 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
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
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[cifar100,h2h,'${CND_TAG}',seed'${SEED}']'

module purge
