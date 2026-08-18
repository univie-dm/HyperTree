#!/bin/bash
#SBATCH --job-name=eval_hier_ti_reval
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_hier_ti_reval_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Authoritative re-eval of HIER TI d=128 under the current eval pipeline, to
# settle the 0.848 (RESULTS.md, 2026-04-30, eval JSONs not in repo) vs 0.457
# (local 2026-05-01 JSONs) hDP_ward discrepancy. Latest embeddings per seed;
# seed 3 has since finished training (e95 replaces the old e35 partial).
# Fresh tags so the 2026-05-01 JSONs are preserved for comparison.

set -e

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb128_eval_90.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_s1_d128_e90_reval0720

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed2_emb128_eval_80.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_s2_d128_e80_reval0720

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed3_emb128_eval_95.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_s3_d128_e95_reval0720

module purge
