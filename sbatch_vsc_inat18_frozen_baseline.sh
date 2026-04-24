#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_frozen_baseline"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=02:00:00
#SBATCH --error=logs/inat18_frozen_baseline_%j.err
#SBATCH --output=logs/inat18_frozen_baseline_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ============================================================================
# iNat18 frozen-DINO baseline — a pure lower-bound reference for the paper.
#
# "What does the frozen self-supervised backbone look like when scored by our
# clustering pipeline, with NO head trained on top?"  This is the zero-effort
# baseline that every hyperbolic-head condition (warm, no_hyphc, curvature
# sweep rescue) should beat.  Useful because SeCu/SCAN don't publish iNat18
# numbers and scaling them to 8142 classes is infeasible, so we need a
# trivial reference to anchor the iNat rows.
#
# Plan:
#   1. Extract frozen FB DINO ResNet50 features for the iNat18 val split
#      (~24k images, 8142 classes).
#   2. Save as (x, y, index) .pt in HIER-compatible format.
#   3. Run scripts/eval_baseline_embeddings.py in Euclidean mode on the
#      saved embeddings → n@1, DP_ward, DP_complete, kmeans NMI/ARI, etc.
#
# Compute: single 1×H100 job, ~30-60 min including extract + eval.
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

# Stage data to local SSD if possible (same pattern as the main iNat sweep)
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

OUT_PT="baselines/frozen_dino/inat18_val_dino_r50.pt"

echo "=== [1/2] extract frozen DINO features on iNat18 val ==="
uv run python scripts/extract_frozen_backbone_embeddings.py \
    --dataset inat18 \
    --ckpt ${DINO_CKPT} \
    --data_dir ${DATA_DIR} \
    --split val \
    --resize 224 \
    --batch_size 256 \
    --num_workers 12 \
    --out ${OUT_PT}

echo "=== [2/2] eval via scripts/eval_baseline_embeddings.py ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${OUT_PT} \
    --manifold euclidean --c 0.0 \
    --dataset inat18 \
    --tag frozen_dino_inat18_val
