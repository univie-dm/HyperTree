#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "imnet_dino_pca"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=04:00:00
#SBATCH --error=logs/imagenet_dino_pca_%j.err
#SBATCH --output=logs/imagenet_dino_pca_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# ============================================================================
# ImageNet matched-dim DINO ceiling — closes the "0.884 vs 32/128 is apples
# to oranges" caveat in RESULTS.md.
#
# Plan:
#   1. Extract frozen FB DINO ResNet50 (2048-d) features for ImageNet val
#      (10k stratified subsample to match our trained-model eval protocol).
#   2. PCA-reduce to {32, 128} dimensions.
#   3. Run eval_baseline_embeddings.py at each reduced dim.
#
# Output: hDP_ward + n@1 at matched dims, so the comparison "our trained
# d=128 vs frozen DINO PCA-128" is finally apples-to-apples.
#
# Single GPU, ~1h total (extract dominates).
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

LOCAL_DATA="/tmp/imagenet"
if [ ! -d "$LOCAL_DATA/val" ] && [ -d "$DATA/imagenet/val" ]; then
    echo "Copying ImageNet val to local SSD..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/imagenet/val" "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA/val" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="$DATA/imagenet"
fi

OUT_2048="baselines/frozen_dino/imagenet_val_dino_r50_2048d.pt"

echo "=== [1/3] extract frozen DINO 2048-d features on ImageNet val ==="
uv run python scripts/extract_frozen_backbone_embeddings.py \
    --dataset imagenet \
    --ckpt ${DINO_CKPT} \
    --data_dir ${DATA_DIR} \
    --split val \
    --resize 224 \
    --batch_size 256 \
    --num_workers 12 \
    --out ${OUT_2048}

for D in 32 128; do
    OUT_PCA="baselines/frozen_dino/imagenet_val_dino_r50_pca${D}d.pt"
    echo "=== [2/3] PCA reduce 2048 -> ${D} ==="
    uv run python scripts/pca_reduce_embeddings.py \
        --in_path ${OUT_2048} \
        --out_path ${OUT_PCA} \
        --dim ${D}

    echo "=== [3/3] eval at d=${D} ==="
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path ${OUT_PCA} \
        --manifold euclidean --c 0.0 \
        --dataset imagenet \
        --tag frozen_dino_imagenet_val_pca${D}
done
