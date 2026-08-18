#!/bin/bash
#SBATCH --job-name=eval_in1k_lh
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -w shelob
#SBATCH --output=logs/eval_in1k_lh_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Pinned to shelob so all arms of this comparison share one machine
# (cross-node float differences move hDP by up to 0.19 via Ward tie-breaking).
EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset imagenet --use_root False --test_sample_size 10000"
for ce in 0 10 20 30 40 50; do
  f=baselines/rebuild/in1k_lhonly_d128_seed42_clustering_ce${ce}.pt
  [ -f "$f" ] && $EV --emb_path "$f" --tag in1k_lhonly_s42_ce${ce} || echo "absent ce${ce}"
done
module purge
