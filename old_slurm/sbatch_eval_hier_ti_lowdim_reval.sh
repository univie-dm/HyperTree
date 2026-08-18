#!/bin/bash
#SBATCH --job-name=eval_hier_ti_lowdim_reval
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_hier_ti_lowdim_reval_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Re-eval of HIER TI d=16/d=32 under the current eval pipeline. The paper's
# 0.447/0.425 came from the same eval batch that produced the stale-pipeline
# d=128 value of 0.457 (current pipeline gives 0.846 there), so the low-d
# numbers must be re-evaluated on the same epochs before the "HIER collapses
# at low d on TI" comparison is safe.

set -e

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb16_lowdim_eval_95.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_d16_lowdim_s1_ep95_reval0720

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb32_lowdim_eval_70.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --tag hier_ti_d32_lowdim_s1_ep70_reval0720

module purge
