#!/bin/bash
#SBATCH --job-name=ti_hctemp
#SBATCH --time=0-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tiny_imagenet_hctemp_%A_%a.log
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
# TinyImageNet HypHC-temperature sweep at the zany-snow recipe.
#
# Fixed (matches zany-snow-1016 / bright-wave-1025 — pretrain hash is shared
# so all 4 tasks load the same cached PoincareBall ResNet50 SimCLR pretrain
# from data/models/<hash>.pt instead of re-pretraining):
#   r50, latent_dim=128, PoincareBall, freeze_encoder=True,
#   pretrain temp=0.5, backbone_knn=False, seed=42, 800 pre + 600 clu.
#
# Varying: HypHC start_temperature, end_temperature, end_weight.
#   0  baseline       st=0.2 et=0.1 ew=5  (= zany-snow control rerun)
#   1  warmer         st=0.5 et=0.3 ew=5
#   2  much_warmer    st=1.0 et=0.5 ew=5
#   3  weaker_warmer  st=0.5 et=0.3 ew=1
# ============================================================================

case $SLURM_ARRAY_TASK_ID in
  0) ST=0.2; ET=0.1; EW=5.0; TAG=baseline       ;;
  1) ST=0.5; ET=0.3; EW=5.0; TAG=warmer         ;;
  2) ST=1.0; ET=0.5; EW=5.0; TAG=much_warmer    ;;
  3) ST=0.5; ET=0.3; EW=1.0; TAG=weaker_warmer  ;;
esac

echo "=== hctemp task ${SLURM_ARRAY_TASK_ID}: ${TAG} (st=${ST} et=${ET} ew=${EW}) ==="

uv run python train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=42 \
    experiment.batch_size=512 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
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
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=${EW} \
    clustering.hyperbolic_loss.start_temperature=${ST} \
    clustering.hyperbolic_loss.end_temperature=${ET} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,hctemp,'${TAG}']'

module purge
