#!/bin/bash
#SBATCH --job-name=eval_hier_ti
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_hier_ti_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Quick eval of HIER TI embeddings at the latest-saved (best-validation) epoch
# per seed. seed=3 is still training; we eval its current best (e=35).

set -e

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb128_eval_90.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_s1_d128_e90

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed2_emb128_eval_80.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_s2_d128_e80

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed3_emb128_eval_35.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_s3_d128_e35_partial

module purge
