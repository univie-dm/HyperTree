#!/bin/bash
#SBATCH --job-name=ti_imgnet
#SBATCH --time=0-06:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tiny_imagenet_imgnet_backbone_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-1%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# TinyImageNet with a *frozen ImageNet-pretrained* ResNet50 backbone at 224x224.
# Skips contrastive pretraining entirely (epochs=0) and only trains the
# hyperbolic head on top of fixed torchvision ImageNet-1k features.
#
# Motivation: TI's flat DP on SimCLR features is low (~0.10) because R@1 is
# only ~0.27 -- the backbone can't retrieve same-class neighbors. TI classes
# are a strict subset of ImageNet-1k, so supervised ImageNet features should
# push R@1 much higher and drag flat DP up. Hierarchical DP is already ~0.5,
# so this is a flat-DP-targeted intervention.
#
# Conditions (same seed, differ only on HypHC end_weight):
#   0  poincare_hyphc       HypHC end_weight=5  -> h2h vs bright-wave
#   1  poincare_no_hyphc    HypHC end_weight=0  -> h2h vs woven-spaceship
#
# Hash differs from existing TI runs (pretrain_epochs=0, load_supervised_weights=True),
# so no cache collision.
# ============================================================================

case $SLURM_ARRAY_TASK_ID in
  0) HC_W=5.0; CND_TAG=poincare_hyphc      ;;
  1) HC_W=0.0; CND_TAG=poincare_no_hyphc   ;;
esac

echo "=== imgnet_backbone task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} (hyphc_end_weight=${HC_W}) ==="

uv run python train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    dataset.resize=224 \
    experiment.seed=42 \
    experiment.batch_size=64 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=True \
    pretraining.augmentation=byol \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=300 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,imgnet_backbone,'${CND_TAG}']'

module purge
