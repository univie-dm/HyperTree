#!/bin/bash
#SBATCH --job-name=eval_null_c100
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_null_c100_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/euclidean_npy_converted/random_c100_test.pt \
    --manifold euclidean --dataset cifar100 \
    --tag random_c100_nulltest
module purge
