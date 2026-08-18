#!/bin/bash
#SBATCH --job-name=pilotC_ti_final
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/pilotC_ti_final_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
# TI verdict under the pinned variant-C protocol: ours (600-ep rerun finals,
# trailing 3, + ce0 anchors) vs SCAN/SeCu (native euclidean). HIER TI (C) and
# the random floor already exist (0.457±0.023 and 0.411).
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
for s in 42 43 44; do
  for ce in 0 580 590 600; do
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path baselines/ti_d128_full600/ti_d128_seed${s}_ce${ce}.pt \
        --dataset tiny_imagenet --use_root False --manifold poincare --c 1.0 \
        --tag pilotCti_ours_s${s}_ce${ce}
  done
done
for s in 7 42 56; do
  uv run python scripts/eval_baseline_embeddings.py \
      --emb_path baselines/euclidean_npy_converted/secu_ti_s${s}_test.pt \
      --dataset tiny_imagenet --use_root False --manifold euclidean \
      --tag pilotCti_secu_s${s}
done
for s in 42 56 99; do
  uv run python scripts/eval_baseline_embeddings.py \
      --emb_path baselines/euclidean_npy_converted/scan_ti_scan_ti_s${s}_test.pt \
      --dataset tiny_imagenet --use_root False --manifold euclidean \
      --tag pilotCti_scan_s${s}
done
module purge
