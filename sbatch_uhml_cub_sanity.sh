#!/bin/bash
#SBATCH --job-name=uhml_cub
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/uhml_cub_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

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
#      hyperparameters (--net BN-Inception, --loss branch, --c 1.0, --K 2).
#      UHML's models registry only knows BN-Inception (the original paper's
#      backbone); 'VGG16-BN' was a misread of their CUB recipe.
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

# CUB needs to live at DataSet/cub_200_2011 in UHML's expected layout.
# Reuse our cached CUB if available.
if [ ! -d "DataSet/cub_200_2011" ] && [ -d "${REPO_ROOT}/data/CUB_200_2011" ]; then
    mkdir -p DataSet
    ln -sfn ${REPO_ROOT}/data/CUB_200_2011 DataSet/cub_200_2011
fi

mkdir -p ${REPO_ROOT}/logs/uhml/cub

echo "=== [2/2] UHML CUB train (BN-Inception, c=1.0, K=2, 600ep) ==="
uv run python train.py \
    --data cub \
    --data_root DataSet \
    --net BN-Inception \
    --loss branch \
    --hyperbolic True \
    --c 1.0 --K 2 --omega 0.1 \
    --dim 512 \
    --lr 1e-5 \
    --batch_size 128 \
    --num_instances 8 \
    --epochs 600 \
    --save_step 50 \
    --save_dir ${REPO_ROOT}/logs/uhml/cub

module purge
