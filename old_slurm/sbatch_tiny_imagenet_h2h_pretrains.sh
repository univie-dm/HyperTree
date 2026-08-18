#!/bin/bash
#SBATCH --job-name=ti_pretrs
#SBATCH --time=0-16:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tiny_imagenet_h2h_pretrains_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-3%4

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Pretrain-only jobs for the h2h-warm sweep.
# Seeds 42 PoincareBall (zany-snow / bright-wave) and 42 Euclidean
# (different-elevator) pretrains already exist; these add seeds 43, 44 for
# both manifolds so the h2h-warm clustering runs can all load from cache.
# clustering.epochs=0 -> trainer.fit with max_epochs=0 -> skip.
# ============================================================================

case $SLURM_ARRAY_TASK_ID in
  0) SEED=43; MANIFOLD=PoincareBall ;;
  1) SEED=44; MANIFOLD=PoincareBall ;;
  2) SEED=43; MANIFOLD=Euclidean    ;;
  3) SEED=44; MANIFOLD=Euclidean    ;;
esac

echo "=== pretrain task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} manifold=${MANIFOLD} ==="

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
    clustering.epochs=0 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,h2h_pretrain,'${MANIFOLD}',seed'${SEED}']'

module purge
