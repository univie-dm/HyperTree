#!/bin/bash
#SBATCH --job-name=c100_eucl_d16
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_eucl_d16_%A_%a.log
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
# CIFAR-100 Euclidean d=16 baseline — matches the Poincaré/Hyperboloid d=16
# recipe used by the *-16-Ours / *-16-Chami sweeps so the paper main table has
# a fair low-dim Euclidean cell. Those reached hDP_ward ≈ 0.901 (Poincaré) and
# 0.874 (Hyperboloid) at d=16. Previously the only Euclidean CIFAR-100 number
# was from the H2H sweep at d=128 / freeze_encoder — not a fair comparison.
#
# Recipe (from wandb run dbmauahs = Poin2-16-Ours):
#   SimCLR 800ep pretrain (lr=3e-3) + clustering 300ep (lr=1e-4)
#   latent_dim=16, projection head [512,256,128]
#   Contrastive enabled DURING clustering (weight=0.1, t=0.5) — no freeze
#   HypHC disabled (end_weight=0) since Euclidean has no hyperbolic LCA
#   Seeds 1, 2, 3 to match the hyperbolic *-16-Ours runs
# ============================================================================

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== c100_eucl_d16 task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

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
    architecture.manifold_name=Euclidean \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    architecture.projection_layer_dims=[512,256,128] \
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
    clustering.learning_rate=0.0001 \
    clustering.freeze_encoder=False \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=0.0 \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,euclidean_d16_ours,paper_baseline,seed'${SEED}']'

module purge
