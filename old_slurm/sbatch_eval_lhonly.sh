#!/bin/bash
#SBATCH --job-name=eval_lhonly
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_lhonly_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
EV="uv run python scripts/eval_baseline_embeddings.py --use_root False --manifold poincare --c 1.0 --dataset cifar100"
for s in 1 2 3; do
  for ce in 0 290 300; do
    f=baselines/rebuild/c100_lhonly_d16_seed${s}_clustering_ce${ce}.pt
    [ -f "$f" ] && $EV --emb_path "$f" --tag lhonly_c100d16_s${s}_ce${ce} || echo "skip $f"
  done
done
