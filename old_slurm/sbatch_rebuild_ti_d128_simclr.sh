#!/bin/bash
#SBATCH --job-name=rb_ti_d128_simclr
#SBATCH --time=1-12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/rb_ti_d128_simclr_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-5%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# TI d=128 on the SimCLR backbone — rebuttal rebuild. Per the decision to
# standardize TinyImageNet on SimCLR (matching the paper's Setup claim), this
# replaces the DINO-backbone tiny_imagenet_dino d=128 rows. Mirrors
# sbatch_tiny_imagenet_h2h_warm.sh's poincare_hyphc arm; the Hyperboloid arm
# swaps the manifold + pins HyperbolicLinearPP (as in the C100 manifold
# ablation). Dumps per-eval embeddings for pinned-protocol evaluation.
# Array: manifold_idx * 3 + seed_idx.
# ============================================================================

SEEDS=(42 43 44)
MANIFOLDS=(PoincareBall Hyperboloid)
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 3))]}
MANIFOLD=${MANIFOLDS[$((SLURM_ARRAY_TASK_ID / 3))]}
case $MANIFOLD in
  PoincareBall) ARM=ti_poin_d128_simclr; EXTRA="" ;;
  Hyperboloid)  ARM=ti_hyp_d128_simclr;  EXTRA="architecture.hyperbolic_encoding=HyperbolicLinearPP" ;;
esac
export HYP_DUMP_EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/rebuild/${ARM}_seed${SEED}

echo "=== rb_ti_d128_simclr task ${SLURM_ARRAY_TASK_ID}: ${MANIFOLD} seed=${SEED} ==="

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
    ${EXTRA} \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=600 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags="[tiny_imagenet,rebuild_simclr,d128,${MANIFOLD},seed${SEED}]"

module purge
