#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "imagenet_frozen_baseline"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=02:00:00
#SBATCH --error=logs/imagenet_frozen_baseline_%j.err
#SBATCH --output=logs/imagenet_frozen_baseline_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ============================================================================
# ImageNet-1k frozen-DINO baseline — pure lower-bound reference for the paper.
#
# Same design as sbatch_vsc_inat18_frozen_baseline.sh but on ImageNet-1k val.
#
# "What does the frozen self-supervised backbone look like when scored by our
# clustering pipeline, with NO head trained on top?"  Every hyperbolic-head
# condition should beat this.  Also a natural anchor alongside SeCu's
# published ImageNet-1k numbers (NMI=0.797 at d=128) — ours will show where
# pure SSL features sit.
#
# Uses the same 10k test_sample_size as the main ImageNet H2H sweep
# (evaluation_metrics.test_sample_size=10000) so numbers are apples-to-apples.
#
# Plan:
#   1. Extract frozen FB DINO ResNet50 features for the ImageNet val split
#      (50k images, 1000 classes).  stripe fc → 2048-D features.
#   2. Save as (x, y, index) .pt in HIER-compatible format.
#   3. Run scripts/eval_baseline_embeddings.py in Euclidean mode
#      with --test_sample_size 10000 (matches main sweep protocol).
#
# Compute: single 1×H100 job, ~30-60 min including extract + eval.
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

# Stage data to local SSD if possible (same pattern as the main ImageNet sweep)
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

OUT_PT="baselines/frozen_dino/imagenet_val_dino_r50.pt"

echo "=== [1/2] extract frozen DINO features on ImageNet val ==="
uv run python scripts/extract_frozen_backbone_embeddings.py \
    --dataset imagenet \
    --ckpt ${DINO_CKPT} \
    --data_dir ${DATA_DIR} \
    --split val \
    --resize 224 \
    --batch_size 256 \
    --num_workers 12 \
    --out ${OUT_PT}

echo "=== [2/2] eval via scripts/eval_baseline_embeddings.py (10k subsample) ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${OUT_PT} \
    --manifold euclidean --c 0.0 \
    --dataset imagenet \
    --test_sample_size 10000 \
    --tag frozen_dino_imagenet_val
