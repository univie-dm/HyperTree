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
#SBATCH --array=0-23%12

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# STL-10 H2H sweep on VSC. STL-10 is small (13k labeled + 100k unlabeled) and
# trains fast, so we can afford a full-grid sweep with 3 seeds.
# Dataset: configs/dataset/stl10.yaml + src/datasets/stl10.py wrapper.
# Hierarchy: 2-level (fine class -> vehicle/animal) from
# _get_class_names_and_coarse -- WordNet hierarchy file not needed, keep
# evaluation_metrics.use_wordnet_labels=False.
# 4 conditions x 2 backbone_knn x 3 seeds = 24 runs.
# Only 6 unique SimCLR pretrains (2 manifolds x 3 seeds) are computed; the
# rest load from cache.
#
# Array decoding:
#   task_id = seed_idx * 8  +  bk_idx * 4  +  cond_idx
#   cond_idx (0..3)  -> see CONDS below
#   bk_idx   (0..1)  -> backbone_knn: False, True
#   seed_idx (0..2)  -> seeds 42, 43, 44
#
# Condition packing: "manifold:end_weight:start_temp:end_temp:tag"
#   0  PoincareBall + HypHC(ew=5) warm schedule (st=1.0, et=0.5)
#   1  PoincareBall + HypHC(ew=5) cold schedule (st=0.2, et=0.1)
#   2  PoincareBall + no HypHC
#   3  Euclidean    + no HypHC  (baseline)
# ============================================================================

CONDS=(
  "PoincareBall:5.0:1.0:0.5:poincare_hyphc_warm"
  "PoincareBall:5.0:0.2:0.1:poincare_hyphc_cold"
  "PoincareBall:0.0:0.2:0.1:poincare_no_hyphc"
  "Euclidean:0.0:0.2:0.1:euclidean_baseline"
)
SEEDS=(42 43 44)
BK_VALS=(False True)

COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))
BK_IDX=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID / 8) % 3 ))

IFS=':' read -r MANIFOLD HC_W ST_T ET_T CND_TAG <<< "${CONDS[$COND_IDX]}"
BK_KNN=${BK_VALS[$BK_IDX]}
SEED=${SEEDS[$SEED_IDX]}
BK_TAG=$([ "$BK_KNN" = "True" ] && echo "bk1" || echo "bk0")

echo "=== stl10_h2h task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} bk=${BK_KNN} ==="

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
    clustering.hyperbolic_loss.start_temperature=${ST_T} \
    clustering.hyperbolic_loss.end_temperature=${ET_T} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=${BK_KNN} \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.use_wordnet_labels=False \
    wandb.tags='[stl10,h2h,vsc,'${CND_TAG}','${BK_TAG}',seed'${SEED}']'
