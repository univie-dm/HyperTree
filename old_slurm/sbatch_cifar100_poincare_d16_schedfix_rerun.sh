#!/bin/bash
#SBATCH --job-name=c100_d16_schedfix_rerun
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_d16_schedfix_rerun_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel,dgx-h100-em2
#SBATCH --array=0-3%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Resubmit of the 4 OOM'd tasks from job 535626. Same recipe, but excludes
# dgx-h100-em2 (shared GPU caused 75GB allocation failures). Only the 4
# missing (seed, cond) combinations are scheduled here; the 2 surviving
# low_weight runs (seed=1, seed=3) are NOT re-run.
#
# task -> (seed, cond)
#   0 -> seed=2, low_weight
#   1 -> seed=1, flat_temp
#   2 -> seed=2, flat_temp
#   3 -> seed=3, flat_temp

case ${SLURM_ARRAY_TASK_ID} in
  0) SEED=2; END_W=1.5; W_INT=20;  END_T=0.1; T_INT=250; CND_TAG=low_weight ;;
  1) SEED=1; END_W=5.0; W_INT=250; END_T=0.2; T_INT=250; CND_TAG=flat_temp  ;;
  2) SEED=2; END_W=5.0; W_INT=250; END_T=0.2; T_INT=250; CND_TAG=flat_temp  ;;
  3) SEED=3; END_W=5.0; W_INT=250; END_T=0.2; T_INT=250; CND_TAG=flat_temp  ;;
esac

echo "=== c100_d16_schedfix_rerun task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ${CND_TAG} END_W=${END_W} W_INT=${W_INT} END_T=${END_T} T_INT=${T_INT} ==="

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
    clustering.hyperbolic_loss.end_weight=${END_W} \
    clustering.hyperbolic_loss.weight_interval=${W_INT} \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=${END_T} \
    clustering.hyperbolic_loss.temperature_interval=${T_INT} \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    architecture.hyperbolic_smoothing=50.0 \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,poincare_d16_schedfix,'${CND_TAG}',seed'${SEED}']'

module purge
