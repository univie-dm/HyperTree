#!/bin/bash
#SBATCH --job-name=frozen_inat
#SBATCH --time=10:00:00
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/frozen_inat_%A.log
#SBATCH -A datamining
#SBATCH -x vader

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# FROZEN-BACKBONE CONTROL on iNat18 val (24,426 images / 8,142 classes).
#
# Answers three reviewer questions with one protocol:
#   * Gy9D-Q1/W1 : does the leaf-level collapse persist with a biology-domain
#                  backbone (BioCLIP) instead of DINO? If it recovers, the
#                  collapse is a backbone limitation, not a HyperTree one.
#   * echd-Q2    : what does the frozen backbone achieve on hDP when the
#                  linkage is applied DIRECTLY to raw features (no head,
#                  no training)?
#   * Y8o7-Q2    : how much of the hierarchy is backbone-driven vs method?
#
# Both backbones share the datamodule's iNat18 test geometry (Resize 224x224,
# no crop) and each gets its own pretraining normalization. A random-feature
# arm gives the noise floor, which is mandatory for reading hDP at all
# (see the 3-variant hDP audit: floors ranged 0.02-0.84).
#
# Evaluated under the PINNED protocol: use_root=False, full val split.
# Features are Euclidean, so --manifold euclidean.
# ============================================================================

OUT=baselines/frozen
mkdir -p ${OUT}

EX="uv run --with open_clip_torch python scripts/extract_backbone_features.py \
    --dataset inat18 --split val --resize 224 --batch_size 256 --num_workers 16"

echo "=== [1/3] extract frozen BioCLIP (ViT-B/16, TreeOfLife-10M) ==="
${EX} --backbone bioclip        --out_pt ${OUT}/inat18_val_bioclip.pt

echo "=== [2/3] extract frozen DINO ResNet-50 (ImageNet-1k SSL) ==="
${EX} --backbone dino_resnet50  --out_pt ${OUT}/inat18_val_dino_rn50.pt

echo "=== [3/3] random-feature floor (matched labels, 512-d) ==="
uv run python -c "
import torch
x,y,i = torch.load('${OUT}/inat18_val_bioclip.pt', weights_only=False)
g = torch.Generator().manual_seed(0)
torch.save((torch.randn(x.shape[0], 512, generator=g), y, i),
           '${OUT}/inat18_val_random.pt')
print('random floor arm:', x.shape[0], 'x 512')
"

EV="uv run python scripts/eval_baseline_embeddings.py \
    --manifold euclidean --c 1.0 --dataset inat18 --use_root False"

echo "=== eval: BioCLIP ==="
${EV} --emb_path ${OUT}/inat18_val_bioclip.pt   --tag frozen_inat_bioclip
echo "=== eval: DINO RN50 ==="
${EV} --emb_path ${OUT}/inat18_val_dino_rn50.pt --tag frozen_inat_dino_rn50
echo "=== eval: random floor ==="
${EV} --emb_path ${OUT}/inat18_val_random.pt    --tag frozen_inat_random

module purge
