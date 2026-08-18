#!/bin/bash
#SBATCH --job-name=pilotC_traj
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/pilotC_traj_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
# Per-epoch hDP trajectories under variant C for the three C100 d16 seeds,
# to see whether the final-epoch seed spread (0.32/0.41/0.57) is a stable
# plateau difference or peak-then-decay timing (schedule overshoot).
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
for s in 1 2 3; do
  for ce in 5 10 15 20 25 30 35 40 45; do
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path baselines/lca_wrong_rate/c100_d16_seed${s}_ce${ce}.pt \
        --dataset cifar100 --use_root False --manifold poincare --c 1.0 \
        --tag pilotC_ours_c100d16_s${s}_ce${ce}
  done
done
module purge
