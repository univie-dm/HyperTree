#!/bin/bash
#SBATCH --job-name=eval_in1k_baselines_ourpipe
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_in1k_baselines_ourpipe_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# SCAN + SeCu ImageNet-1k baseline embeddings evaluated through OUR current
# pipeline: same script, same imagenet hierarchy file, same hrc hDP variant,
# and the paper's 10k class-stratified subsample — replacing the SeCu-repo
# evals (uniform 50k, different imagenet hierarchy file, pre-hrc hDP).

set -e

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/euclidean_npy_converted/scan_in1k_released_test.pt \
    --manifold euclidean --dataset imagenet \
    --test_sample_size 10000 \
    --tag scan_in1k_released_ourpipe_10k

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/euclidean_npy_converted/secu_in1k_s42_test.pt \
    --manifold euclidean --dataset imagenet \
    --test_sample_size 10000 \
    --tag secu_in1k_s42_ourpipe_10k

module purge
