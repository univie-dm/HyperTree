#!/bin/bash
#SBATCH --job-name=scan_ti_evals
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/scan_ti_evals_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
for s in 42 56 99; do
  uv run python scripts/eval_baseline_embeddings.py \
      --emb_path baselines/euclidean_npy_converted/scan_ti_scan_ti_s${s}_test.pt \
      --dataset tiny_imagenet --use_root False --manifold euclidean \
      --tag pilotCti_scan_s${s}
done
