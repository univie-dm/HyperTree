#!/bin/bash
#SBATCH --job-name=ti_h2h
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tiny_imagenet_h2h_%A_%a.log
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
# TinyImageNet hyperbolic-vs-Euclidean head-to-head at the zany-snow recipe.
# Fixed: r50, latent_dim=128, freeze_encoder=True, pretrain_temp=0.5,
#        backbone_knn=False, seed=42, SimCLR pretrain 800ep + clustering 600ep.
#
# Conditions (differ only on manifold + HypHC weight):
#   0  poincare_hyphc       PoincareBall + HypHC end_weight=5   (= zany-snow spec)
#   1  poincare_no_hyphc    PoincareBall + HypHC end_weight=0   (hyp embed, no HC loss)
#   2  euclidean_baseline   Euclidean    + HypHC end_weight=0   (plain Euclidean)
#
# Three distinct (architecture, seed) hashes -> three independent pretrains,
# no cache races. Runs sequentially-parallel with %3.
# ============================================================================

case $SLURM_ARRAY_TASK_ID in
  0) MANIFOLD=PoincareBall; HC_W=5.0; CND_TAG=poincare_hyphc      ;;
  1) MANIFOLD=PoincareBall; HC_W=0.0; CND_TAG=poincare_no_hyphc   ;;
  2) MANIFOLD=Euclidean;    HC_W=0.0; CND_TAG=euclidean_baseline  ;;
esac

echo "=== h2h task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} ==="
echo "manifold=${MANIFOLD} | hyphc_end_weight=${HC_W}"

uv run python train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=42 \
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
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,h2h,'${CND_TAG}']'

module purge