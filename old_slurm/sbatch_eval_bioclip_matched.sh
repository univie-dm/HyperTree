#!/bin/bash
#SBATCH --job-name=eval_bioclip_m
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_bioclip_m_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Match Table 10's evaluation exactly: 10k class-stratified subsample, Poincare
# c=0.3, use_root=False. The full-split numbers are NOT comparable to the
# paper's iNat rows, which use test_sample_size=10000 over 8142 classes.
EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 0.3 --dataset inat18 --use_root False --test_sample_size 10000"
for s in 1 2 3; do
  for ce in 0 10 50; do
    f=baselines/inat_bioclip/inat_bioclip_seed${s}_clustering_ce${ce}.pt
    [ -f "$f" ] && $EV --emb_path "$f" --tag bioclipM_s${s}_ce${ce}
  done
done
module purge
