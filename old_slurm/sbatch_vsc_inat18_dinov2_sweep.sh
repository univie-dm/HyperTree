#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_dinov2_sweep"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=06:00:00
#SBATCH --error=logs/inat18_dinov2_sweep_%A_%a.err
#SBATCH --output=logs/inat18_dinov2_sweep_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-4

set -euo pipefail

# ============================================================================
# DINOv2 frozen iNat18 sweep — push the baseline above the 0.851 hDP_ward
# from sbatch_vsc_inat18_dinov2_frozen.sh (job 1170241, ViT-L/14 @ 224).
#
# Two knobs with strong priors for fine-grained iNat:
#   - Resolution: ViT was trained at 224 but iNat species discrimination
#     should benefit from higher input res (more tokens, more detail).
#   - Backbone size: ViT-G/14 (1536-d, ~3× params) over ViT-L/14 (1024-d).
#
# Configs (5 array tasks, parallel on H100):
#   0: vitl14 @ 336   — same model, +50% res
#   1: vitl14 @ 448   — same model, 2× res
#   2: vitg14 @ 224   — biggest model, baseline res (isolate model effect)
#   3: vitg14 @ 336   — biggest model, +50% res
#   4: vitg14 @ 448   — biggest model + max res (best-case)
#
# Batch sizes scaled down for higher res / bigger model to fit 80GB H100.
# Eval cost dominates extract for ViT-L; for ViT-G/448 the extract may
# match eval — 6h budget covers worst case.
# ============================================================================

CONFIGS=(
    "dinov2_vitl14 336 96"
    "dinov2_vitl14 448 48"
    "dinov2_vitg14 224 96"
    "dinov2_vitg14 336 48"
    "dinov2_vitg14 448 24"
)
read BACKBONE RES BS <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"

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

TAG="${BACKBONE}_res${RES}"
OUT_RAW="baselines/frozen_dinov2/inat18_val_${TAG}.pt"
HIER="./data/inat18_taxonomy_hierarchy.txt"

if [ ! -f "${HIER}" ]; then
    echo "ERROR: ${HIER} not found. Run scripts/build_inat18_taxonomy.py first."
    exit 1
fi

echo "=== sweep config: backbone=${BACKBONE} res=${RES} bs=${BS} ==="

echo "=== [1/4] extract ${BACKBONE} (res ${RES}) on iNat18 val ==="
uv run python scripts/extract_frozen_backbone_embeddings.py \
    --dataset inat18 \
    --backbone ${BACKBONE} \
    --data_dir ${DATA_DIR} \
    --split val \
    --resize ${RES} \
    --batch_size ${BS} \
    --num_workers 12 \
    --out ${OUT_RAW}

echo "=== [2/4] eval frozen raw-dim with iNat hierarchy ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${OUT_RAW} \
    --manifold euclidean --c 0.0 \
    --dataset inat18 \
    --wordnet_path ${HIER} \
    --tag frozen_${TAG}_raw

for D in 128 256; do
    OUT_PCA="baselines/frozen_dinov2/inat18_val_${TAG}_pca${D}d.pt"
    echo "=== [3/4] PCA reduce ${TAG} -> ${D} ==="
    uv run python scripts/pca_reduce_embeddings.py \
        --in_path ${OUT_RAW} \
        --out_path ${OUT_PCA} \
        --dim ${D}

    echo "=== [4/4] eval matched-dim ${TAG} d=${D} ==="
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path ${OUT_PCA} \
        --manifold euclidean --c 0.0 \
        --dataset inat18 \
        --wordnet_path ${HIER} \
        --tag frozen_${TAG}_pca${D}
done
