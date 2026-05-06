#!/bin/bash
#SBATCH --job-name=ti_h2h_w
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tiny_imagenet_h2h_warm_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-8

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# TinyImageNet h2h at much_warmer HypHC temperature (st=1.0, et=0.5), the
# winning config from the hctemp sweep (cool-music-1032 hit hDP_ward=0.731).
# 3 conditions x 3 seeds = 9 runs. Pretrains expected to exist in cache
# (seeds 42, 43, 44 on both PoincareBall and Euclidean). Clustering only.
#
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

echo "=== h2h_warm task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ${CND_TAG} ==="

uv run python train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
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
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,h2h_warm,'${CND_TAG}',seed'${SEED}']'

module purge
