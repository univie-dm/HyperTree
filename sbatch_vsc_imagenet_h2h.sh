#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "imagenet_h2h"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-12:00:00
#SBATCH --error=logs/imagenet_h2h_%A_%a.err
#SBATCH --output=logs/imagenet_h2h_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-15%4

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# ImageNet-1k H2H sweep on VSC — frozen torchvision-supervised ResNet50.
#
# Why frozen-pretrained (not from-scratch SSL):
#   - From-scratch DINO on ImageNet-1k is ~1 week on 4xH100; breaks VSC's
#     3-day wallclock.
#   - TI imgnet-backbone partial runs (olive-glitter-1045, ep181/300 clustering)
#     already hit hDP=0.619, so the recipe works -- just needs full runtime.
#   - ImageNet-1k has a 12-level WordNet hierarchy: best fit for the
#     hyperbolic-wins-on-deep-hierarchies story.
#
# Caveat: torchvision weights are *supervised*, not SSL. This is NOT a fully
# unsupervised setup. For a real SSL comparison, add a DINO-checkpoint loader
# (src/models/resnet.py:80 currently only supports weights='DEFAULT') and
# rerun with FB's released DINO ResNet50 checkpoint.
#
# 4 conditions x 2 backbone_knn x 2 seeds = 16 clustering-only runs.
# No pretraining (pretraining.epochs=0 + load_supervised_weights=True).
#
# Array decoding:
#   task_id = seed_idx * 8  +  bk_idx * 4  +  cond_idx
#   cond_idx (0..3)  -> CONDS below
#   bk_idx   (0..1)  -> backbone_knn: False, True
#   seed_idx (0..1)  -> seeds 42, 43
# ============================================================================

CONDS=(
  "PoincareBall:5.0:1.0:0.5:poincare_hyphc_warm"
  "PoincareBall:5.0:0.2:0.1:poincare_hyphc_cold"
  "PoincareBall:0.0:0.2:0.1:poincare_no_hyphc"
  "Euclidean:0.0:0.2:0.1:euclidean_baseline"
)
SEEDS=(42 43)
BK_VALS=(False True)

COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))
BK_IDX=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID / 8) % 2 ))

IFS=':' read -r MANIFOLD HC_W ST_T ET_T CND_TAG <<< "${CONDS[$COND_IDX]}"
BK_KNN=${BK_VALS[$BK_IDX]}
SEED=${SEEDS[$SEED_IDX]}
BK_TAG=$([ "$BK_KNN" = "True" ] && echo "bk1" || echo "bk0")

echo "=== imagenet_h2h task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} bk=${BK_KNN} ==="

LOCAL_DATA="/tmp/imagenet"
if [ ! -d "$LOCAL_DATA/train" ] && [ -d "$DATA/imagenet" ]; then
    echo "Copying ImageNet to local SSD..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/imagenet/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA/train" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="$DATA/imagenet"
fi

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
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
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=True \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.accumulate_gradients=2 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=${ST_T} \
    clustering.hyperbolic_loss.end_temperature=${ET_T} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=${BK_KNN} \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,h2h,vsc,frozen_sup,'${CND_TAG}','${BK_TAG}',seed'${SEED}']'
