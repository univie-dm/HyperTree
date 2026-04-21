#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_h2h"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=3-00:00:00
#SBATCH --error=logs/inat18_h2h_%A_%a.err
#SBATCH --output=logs/inat18_h2h_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-15%8

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 H2H sweep on VSC (zen4_0768_h100x4, 4xH100).
# 4 conditions x 2 backbone_knn x 2 seeds = 16 clustering runs.
# Only 4 unique DINO pretrains (2 manifolds x 2 seeds) are computed; the rest
# load from cache via matching hash (pretraining.overwrite=False).
#
# Array decoding:
#   task_id = seed_idx * 8  +  bk_idx * 4  +  cond_idx
#   cond_idx (0..3)  -> see CONDS below
#   bk_idx   (0..1)  -> backbone_knn: False, True
#   seed_idx (0..1)  -> seeds 42, 43
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
SEEDS=(42 43)
BK_VALS=(False True)

COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))
BK_IDX=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID / 8) % 2 ))

IFS=':' read -r MANIFOLD HC_W ST_T ET_T CND_TAG <<< "${CONDS[$COND_IDX]}"
BK_KNN=${BK_VALS[$BK_IDX]}
SEED=${SEEDS[$SEED_IDX]}
BK_TAG=$([ "$BK_KNN" = "True" ] && echo "bk1" || echo "bk0")

echo "=== inat18_h2h task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} bk=${BK_KNN} ==="

# Stage data to local SSD if possible (adjust for your VSC setup)
LOCAL_DATA="/tmp/inat18"
if [ ! -d "$LOCAL_DATA" ] && [ -d "$DATA/inat18" ]; then
    echo "Copying iNat18 to local SSD..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/inat18/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="data/inat18"
fi

uv run python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=${SEED} \
    experiment.num_workers=6 \
    architecture.manifold_name=${MANIFOLD} \
    architecture.latent_dim=128 \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=dino \
    pretraining.epochs=100 \
    pretraining.overwrite=False \
    pretraining.accumulate_gradients=4 \
    pretraining.enable_checkpointing=True \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.accumulate_gradients=4 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=${ST_T} \
    clustering.hyperbolic_loss.end_temperature=${ET_T} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=${BK_KNN} \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=0 \
    wandb.tags='[inat18,h2h,vsc,'${CND_TAG}','${BK_TAG}',seed'${SEED}']'
