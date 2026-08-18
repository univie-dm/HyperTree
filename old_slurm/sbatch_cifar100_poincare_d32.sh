#!/bin/bash
#SBATCH --job-name=c100_poinc_d32
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_poinc_d32_%A_%a.log
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
# CIFAR-100 PoincareBall d=32 — matched-dim comparison to SeCu-32
# (hDP_ward=0.420 per memory). Same recipe as sbatch_cifar100_poincare_d16.sh
# with latent_dim=32.  Memory flags d=32 as SeCu's sweet spot (beats SeCu-16
# on every Sec 5.3/5.4 metric), so this is the dim we most need our own
# matched numbers on.
#
# Seeds 1, 2, 3 to align with the d=16 Poincaré / Euclidean runs.
# ============================================================================

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== c100_poinc_d32 task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    experiment.seed=${SEED} \
    experiment.batch_size=1024 \
    experiment.log_interval=10 \
    experiment.gradient_clip_value=5 \
    experiment.optimizer=Adam \
    architecture.model_type=resnet50 \
    architecture.latent_dim=32 \
    architecture.manifold_name=PoincareBall \
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
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    architecture.hyperbolic_smoothing=50.0 \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,poincare_d32_ours,paper_main,seed'${SEED}']'

module purge
