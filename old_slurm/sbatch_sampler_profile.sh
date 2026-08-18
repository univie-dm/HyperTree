#!/bin/bash
#SBATCH --job-name=sampler_prof
#SBATCH --time=6:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/sampler_prof_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-3%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# SAMPLER WALL-CLOCK PROFILING (Thomas, AC-2 table + echd-Q3).
#
# Fills the "[x ms (y % of step)]" column of the sampler table: how long the
# importance-rejection loop takes per training step, and what fraction of the
# step that is. HYP_PROFILE_SAMPLER=1 makes hyphc.py emit one
# "PROF sampler_ms=... step_ms=..." line per step; the rejection loop is timed
# on its own, with CUDA synchronisation on both sides so the number is real
# device time and not async launch time.
#
# Only a few steps are needed, so each arm runs a single clustering epoch on
# its dataset at the paper's batch size and recipe. Summarise afterwards with
#   uv run python scripts/report_sampler_profile.py logs/sampler_prof_<jobid>_*.log
# which discards the first 5 steps as warm-up.
#
# Arms: 0 CIFAR-100, 1 TinyImageNet, 2 ImageNet-1k, 3 iNaturalist-2018.
# ============================================================================

export HYP_PROFILE_SAMPLER=1

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"

case $SLURM_ARRAY_TASK_ID in
  0) DS=cifar100;      CFG=config;                BS=1024; DIM=16;  EXTRA="" ;;
  1) DS=tiny_imagenet; CFG=tiny_imagenet_simclr;  BS=512;  DIM=16;  EXTRA="" ;;
  2) DS=imagenet;      CFG=imagenet_dino;         BS=256;  DIM=128; EXTRA="pretraining.pretrained_weights_path=${DINO_CKPT}" ;;
  3) DS=inat18;        CFG=inat18_dino;           BS=256;  DIM=128; EXTRA="pretraining.pretrained_weights_path=${DINO_CKPT} dataset.data_dir=data/inat18" ;;
esac

echo "=== sampler_prof: dataset=${DS} batch_size=${BS} d=${DIM} ==="

uv run python train.py --config-name ${CFG} \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=${DS} \
    experiment.batch_size=${BS} \
    experiment.seed=1 \
    architecture.latent_dim=${DIM} \
    architecture.manifold_name=PoincareBall \
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    clustering.epochs=1 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.sampling_strategy=importance \
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=1.0 \
    evaluation_metrics.visualization=False \
    ${EXTRA}

module purge
