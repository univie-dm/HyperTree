#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "stl10_h2h"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/stl10_h2h_%A_%a.err
#SBATCH --output=logs/stl10_h2h_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-5%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# !! PREREQUISITE -- STL-10 IS NOT YET WIRED INTO THE CODEBASE !!
#
# Before this script can run, the following must be added:
#   1. configs/dataset/stl10.yaml  (resize=96, channels=3, data_dir=...)
#   2. src/datasets/datamodule.py:
#        - Add "stl10" to the Literal in __init__ (line ~227)
#        - Map "stl10" -> torchvision.datasets.STL10 in _DATASETS (line ~248)
#        - STL-10 uses split='train'/'test'/'unlabeled' instead of train=True/False
#   3. data/stl10_hierarchy.txt if we want WordNet hierarchical DP; otherwise
#      evaluation_metrics.use_wordnet_labels=False and rely on flat DP.
#
# STL-10 is chosen here because: (a) small (13k labeled, 100k unlabeled),
# (b) classic SSL benchmark, (c) 96x96 native resolution -- matches DINO
# local_crop_size well. The 100k unlabeled pool is great for DINO pretraining.
#
# H2H conditions (3 conditions x 2 seeds = 6 runs):
#   task_idx = seed_idx*3 + cond_idx
#   seeds 42, 43; conds poincare_hyphc / poincare_no_hyphc / euclidean_baseline
# ============================================================================

SEEDS=(42 43)
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_IDX]}

case $COND_IDX in
  0) MANIFOLD=PoincareBall; HC_W=5.0; CND_TAG=poincare_hyphc      ;;
  1) MANIFOLD=PoincareBall; HC_W=0.0; CND_TAG=poincare_no_hyphc   ;;
  2) MANIFOLD=Euclidean;    HC_W=0.0; CND_TAG=euclidean_baseline  ;;
esac

echo "=== stl10_h2h task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ${CND_TAG} ==="

# Stage data to local SSD if possible
LOCAL_DATA="/tmp/stl10"
if [ ! -d "$LOCAL_DATA" ] && [ -d "$DATA/stl10" ]; then
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/stl10/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="data/stl10"
fi

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=stl10 \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
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
    clustering.epochs=400 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.use_wordnet_labels=False \
    wandb.tags='[stl10,h2h,vsc,'${CND_TAG}',seed'${SEED}']'
