#!/bin/bash
#SBATCH --job-name=lca_wrong_rate
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -w galadriel
#SBATCH --output=logs/lca_wrong_rate_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# §4.2.3 LCA rectification rate: percentage of point pairs where Chami's standard
# LCA formula produces a geometrically-wrong LCA (lies outside the geodesic
# segment) and our rectified version corrects it. Measured on Poincaré
# embeddings from HIER (CVPR 2023) which is a representative trained
# Poincaré-DML embedding on CIFAR-100 / TinyImageNet at c=0.1.
#
# 3 seeds × 2 datasets = 6 measurements + 1 uniform-random baseline.

# CIFAR-100 d=16 (HIER's lowest dim, 3 seeds, peak eval epochs)
uv run python scripts/measure_lca_wrong_rate.py \
    --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb16_eval_75.pt \
    --c 0.1 --tag lca_wrong_hier_c100_s1_d16

uv run python scripts/measure_lca_wrong_rate.py \
    --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed2_emb16_eval_90.pt \
    --c 0.1 --tag lca_wrong_hier_c100_s2_d16

uv run python scripts/measure_lca_wrong_rate.py \
    --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed3_emb16_eval_95.pt \
    --c 0.1 --tag lca_wrong_hier_c100_s3_d16

# TinyImageNet d=128 (3 seeds, latest saved epoch per seed)
uv run python scripts/measure_lca_wrong_rate.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb128_eval_90.pt \
    --c 0.1 --tag lca_wrong_hier_ti_s1_d128

uv run python scripts/measure_lca_wrong_rate.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed2_emb128_eval_80.pt \
    --c 0.1 --tag lca_wrong_hier_ti_s2_d128

uv run python scripts/measure_lca_wrong_rate.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed3_emb128_eval_35.pt \
    --c 0.1 --tag lca_wrong_hier_ti_s3_d128

module purge
