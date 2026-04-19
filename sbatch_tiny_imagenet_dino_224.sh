#!/bin/bash
#SBATCH --job-name=ti_dino224
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tiny_imagenet_dino_224_%A_%a.log
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
# DINO self-supervised pretraining on TinyImageNet *upsampled to 224x224*.
# Prior DINO-on-TI runs at 64x64 starved the multi-crop objective: a global
# crop of 64 and local crop of 32 gives a ratio of 2, far from the standard
# 224/96 ratio that makes multi-crop work. Result: R@1 ~0.14 (half of SimCLR)
# but hierarchical DP ~0.85 (1.7x better than SimCLR's 0.49). Coarse structure
# is there but fine discrimination is destroyed.
#
# This run upsamples TI to 224x224 and uses the standard DINO local_crop_size=96,
# which restores the multi-crop ratio. Goal: keep the coarse/hierarchy advantage
# while recovering fine discrimination.
#
# Fully unsupervised (DINO has no label access), in-domain pretraining,
# hyperbolic head on top. Paper story if this works: unsupervised hyperbolic
# clustering on fine-grained TI recovers hierarchy at R@1 > 0.30 and hDP > 0.80.
#
# Conditions (same seed, differ on HypHC end_weight):
#   0  poincare_hyphc       HypHC end_weight=5
#   1  poincare_no_hyphc    HypHC end_weight=0   (h2h against the condition above)
# ============================================================================

case $SLURM_ARRAY_TASK_ID in
  0) HC_W=5.0; CND_TAG=poincare_hyphc      ;;
  1) HC_W=0.0; CND_TAG=poincare_no_hyphc   ;;
esac

echo "=== dino_224 task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} (hyphc_end_weight=${HC_W}) ==="

uv run python train.py --config-name tiny_imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    dataset.resize=224 \
    experiment.seed=42 \
    experiment.batch_size=128 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=dino \
    pretraining.epochs=100 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=False \
    pretraining.augmentation=dino \
    pretraining.accumulate_gradients=8 \
    pretraining.n_local_crops=6 \
    pretraining.local_crop_size=96 \
    clustering.epochs=300 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,dino,res224,'${CND_TAG}']'

module purge
