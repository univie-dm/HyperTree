#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_dinov2_frozen"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=02:00:00
#SBATCH --error=logs/inat18_dinov2_frozen_%j.err
#SBATCH --output=logs/inat18_dinov2_frozen_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ============================================================================
# iNat18 DINOv2 ViT-L/14 frozen baseline — Sweep B Phase 1.
#
# Hypothesis: DINO ResNet50 was self-supervised on ImageNet (1000 classes),
# never asked to discriminate at species level. Diagnosis from c=0.3 rescue
# trajectory (2026-04-28): n@1 pinned at 0.000164 = 4/24426, NMI even
# decreases during training, hyperbolic_loss decoupled from test metric.
# The bottleneck looks like backbone information content, not the head.
#
# DINOv2 ViT-L/14 was trained on a 142M-image curated dataset that includes
# iNaturalist data (per the DINOv2 paper, "LVD-142M" oversamples natural
# images). Likely already encodes species structure → the frozen ce=0
# anchor on iNat with this backbone is the cheapest test of whether
# better features fix iNat.
#
# Plan:
#   1. Extract DINOv2 ViT-L/14 (1024-d) features for iNat18 val (~24k imgs).
#   2. PCA-reduce 1024 -> {128, 256} for matched-dim eval against our
#      trained head dims.
#   3. Run scripts/eval_baseline_embeddings.py at each dim → hDP_ward,
#      DP_ward, n@1, NMI, all using our standard pipeline + iNat taxonomy.
#
# Compute: single 1×H100, ~30-60 min (ViT-L/14 forward on 24k images).
# Network: torch.hub fetches the DINOv2 weights on first call (~1.2 GB).
# ============================================================================

# Stage data to local SSD if possible (same pattern as the other iNat sbatches)
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

OUT_1024="baselines/frozen_dinov2/inat18_val_dinov2_vitl14.pt"
HIER="./data/inat18_taxonomy_hierarchy.txt"

if [ ! -f "${HIER}" ]; then
    echo "ERROR: ${HIER} not found. Run scripts/build_inat18_taxonomy.py first."
    exit 1
fi

echo "=== [1/4] extract DINOv2 ViT-L/14 (1024-d) features on iNat18 val ==="
uv run python scripts/extract_frozen_backbone_embeddings.py \
    --dataset inat18 \
    --backbone dinov2_vitl14 \
    --data_dir ${DATA_DIR} \
    --split val \
    --resize 224 \
    --batch_size 128 \
    --num_workers 12 \
    --out ${OUT_1024}

echo "=== [2/4] eval frozen 1024-d with iNat hierarchy ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${OUT_1024} \
    --manifold euclidean --c 0.0 \
    --dataset inat18 \
    --wordnet_path ${HIER} \
    --tag frozen_dinov2_inat18_val_1024d

for D in 128 256; do
    OUT_PCA="baselines/frozen_dinov2/inat18_val_dinov2_vitl14_pca${D}d.pt"
    echo "=== [3/4] PCA reduce iNat DINOv2 1024 -> ${D} ==="
    uv run python scripts/pca_reduce_embeddings.py \
        --in_path ${OUT_1024} \
        --out_path ${OUT_PCA} \
        --dim ${D}

    echo "=== [4/4] eval matched-dim DINOv2 d=${D} ==="
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path ${OUT_PCA} \
        --manifold euclidean --c 0.0 \
        --dataset inat18 \
        --wordnet_path ${HIER} \
        --tag frozen_dinov2_inat18_val_pca${D}
done
