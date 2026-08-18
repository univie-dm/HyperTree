#!/bin/bash
#SBATCH --job-name=uhml_cub
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/uhml_cub_%j.log
#SBATCH -A datamining
#SBATCH -w galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# UHML (Yan et al. CVPR 2021, "Unsupervised Hyperbolic Metric Learning") —
# CUB sanity run.
#
# Repo: https://github.com/JiexiYan/UnHyperML
#
# This sbatch:
#   1. Clones the repo to baselines/UnHyperML/ (idempotent — re-runs no-op)
#   2. Runs UHML's stock CUB training recipe with their default
#      hyperparameters (--net BN-Inception, --loss Triplet, --c 1.0, --K 2).
#      UHML's losses registry has only {NCA, Contrastive, Binomial,
#      LiftedStructure, Triplet, LogRatio} — the upstream default
#      `--loss branch` (the multi-task branch loss from the paper) is
#      not committed in their public repo. Triplet is the closest
#      registered loss to UHML's hard-mined-triplet approach. If this
#      lands, we have a working UHML smoke; we can then either patch
#      a BranchLoss class or report Triplet-loss numbers as the UHML
#      row in §5.3.
#   3. Saves checkpoints to logs/uhml/cub/
#
# Purpose: prove the UHML codebase trains in our environment before doing
# the (substantial) work of writing adapters for CIFAR-100 / TinyImageNet /
# ImageNet / iNat18 — UHML only ships with CUB / Cars / SOP adapters.
#
# Once this run lands, the integration plan for §5.3 UHML rows is:
#   a. Write baselines/UnHyperML/DataSet/{cifar100,tiny_imagenet,imagenet,
#      inat18}.py mirroring their existing CUB adapter.
#   b. Run training per dataset; save embeddings via Model2Feature.py.
#   c. Pipe through scripts/eval_baseline_embeddings.py for matched metrics.
# ============================================================================

REPO_ROOT=/srv/home/users/sidakk95cs/hyperbolic-clustering
BASE=${REPO_ROOT}/baselines/UnHyperML

cd ${REPO_ROOT}/baselines

if [ ! -d "UnHyperML/.git" ]; then
    echo "=== [1/2] cloning UnHyperML ==="
    git clone https://github.com/JiexiYan/UnHyperML.git
fi
cd UnHyperML

# UHML expects DataSet/CUB_200_2011/train.txt (capital path, image-path/class-id pairs).
# 1) symlink the dataset under the capital-case name UHML expects
# 2) build train.txt from images.txt + image_class_labels.txt (UHML uses the
#    full set as training; eval is via clustering on extracted features).
mkdir -p DataSet
if [ ! -e "DataSet/CUB_200_2011" ] && [ -d "${REPO_ROOT}/data/CUB_200_2011" ]; then
    ln -sfn ${REPO_ROOT}/data/CUB_200_2011 DataSet/CUB_200_2011_link
    # symlink to the parent so that DataSet/CUB_200_2011/{images,images.txt,...} resolves
    rmdir DataSet 2>/dev/null || true
    mkdir -p DataSet
    ln -sfn ${REPO_ROOT}/data/CUB_200_2011 DataSet/CUB_200_2011
fi
# Generate train.txt and test.txt using the standard CUB DML class split
# (classes 1-100 = train, 101-200 = test, no class overlap — Song et al.).
# Paths are relative to DataSet/CUB_200_2011/images/, labels are 0-indexed.
TRAIN_TXT="DataSet/CUB_200_2011/train.txt"
TEST_TXT="DataSet/CUB_200_2011/test.txt"
if [ ! -f "$TRAIN_TXT" ] || [ ! -f "$TEST_TXT" ]; then
    uv run python -c "
from pathlib import Path
root = Path('${REPO_ROOT}/data/CUB_200_2011')
ids_to_path = dict(line.strip().split(' ', 1) for line in open(root/'images.txt'))
ids_to_label = {k: int(v) for k, v in (line.strip().split(' ', 1) for line in open(root/'image_class_labels.txt'))}
with open('$TRAIN_TXT', 'w') as ftr, open('$TEST_TXT', 'w') as fte:
    for i in sorted(ids_to_path, key=int):
        cls = ids_to_label[i]
        line = f'images/{ids_to_path[i]} {cls - 1}\n'
        (ftr if cls <= 100 else fte).write(line)
print('train.txt:', sum(1 for _ in open('$TRAIN_TXT')), 'lines')
print('test.txt :', sum(1 for _ in open('$TEST_TXT')), 'lines')
"
fi

mkdir -p ${REPO_ROOT}/logs/uhml/cub

echo "=== [2/2] UHML CUB train (BN-Inception, Triplet, c=1.0, K=2, 600ep, lr=1e-4) ==="
uv run python train.py \
    --data cub \
    --data_root DataSet \
    --net BN-Inception \
    --loss Triplet \
    --hyperbolic True \
    --c 1.0 --K 2 --omega 0.1 \
    --dim 512 \
    --lr 1e-4 \
    --batch_size 128 \
    --num_instances 8 \
    --epochs 600 \
    --save_step 50 \
    --save_dir ${REPO_ROOT}/logs/uhml/cub

module purge
