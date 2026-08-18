#!/bin/bash
#SBATCH --job-name=c100_d16_schedfix
#SBATCH --time=1-00:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_d16_schedfix_%A_%a.log
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
# CIFAR-100 d=16 Poincaré: HypHC schedule fix.
#
# Diagnosis from worldly-oath-1224 (current paper_main run): hDP_ward
# trajectory shows a SPIKE at clustering ep 11-25 (hDP ~0.90, matching the
# historical 0.901), then crashes when norm_max crosses ~0.995 (boundary
# saturation). End_weight=5.0 + end_temp=0.1 pushes too hard.
#
# This sweep tests two knobs separately (so we know which is the problem):
#   COND 0 "low_weight":  end_weight = 1.5, weight_interval = 20 (ramp 1->1.5
#                          in 20 epochs then hold; never reaches saturating
#                          weight). Temp schedule unchanged (0.2 -> 0.1).
#   COND 1 "flat_temp":   end_temperature = 0.2 (no temp ramp). Weight
#                          schedule unchanged (1.0 -> 5.0).
#
# task_idx = seed_idx * 2 + cond_idx
#   seed_idx 0,1,2 = seeds 1, 2, 3
#   cond_idx 0 = low_weight,  1 = flat_temp
#
# Reuses cached SimCLR pretrained encoder (overwrite=False) — same hash as
# the worldly-oath-1224 baseline so the comparison is direct.
# ============================================================================

SEEDS=(1 2 3)
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEED=${SEEDS[$SEED_IDX]}

case $COND_IDX in
  0) END_W=1.5;  W_INT=20;   END_T=0.1; T_INT=250; CND_TAG=low_weight ;;
  1) END_W=5.0;  W_INT=250;  END_T=0.2; T_INT=250; CND_TAG=flat_temp  ;;
esac

echo "=== c100_d16_schedfix task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ${CND_TAG} END_W=${END_W} W_INT=${W_INT} END_T=${END_T} T_INT=${T_INT} ==="

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
