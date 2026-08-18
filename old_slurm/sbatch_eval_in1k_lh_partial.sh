#!/bin/bash
#SBATCH --job-name=eval_in1k_lhp
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_in1k_lhp_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset imagenet --use_root False --test_sample_size 10000"
for ce in 0 10 20 30 40 50; do
  for pre in in1k_lhonly1g in1k_lhonly; do
    f=baselines/rebuild/${pre}_d128_seed42_clustering_ce${ce}.pt
    [ -f "$f" ] && $EV --emb_path "$f" --tag ${pre}_s42_ce${ce}
  done
done
module purge
