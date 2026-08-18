#!/bin/bash
#SBATCH --job-name=eval_hdp_null_test
#SBATCH --time=0:45:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_hdp_null_test_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Null test for the current hDP implementation: random gaussian embeddings
# with real label structure. If random scores high (~0.8+), the current hDP
# variant is saturated/broken and its absolute numbers are uninformative.

set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/euclidean_npy_converted/random_in1k_test.pt \
    --manifold euclidean --dataset imagenet \
    --test_sample_size 10000 \
    --tag random_in1k_nulltest

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/euclidean_npy_converted/random_ti_test.pt \
    --manifold euclidean --dataset tiny_imagenet \
    --tag random_ti_nulltest

module purge
