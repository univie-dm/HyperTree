#!/bin/bash
#SBATCH --job-name=eval_bioclip
#SBATCH --time=4:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_bioclip_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Gy9D Q1 Phase 2: HyperTree head on frozen BioCLIP, iNat18, pinned protocol.
# Compared against this run's own ce=0 anchor and the Phase-1 frozen controls.
EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 0.3 --dataset inat18 --use_root False"
for s in 1 2 3; do
  for ce in 0 50; do
    f=baselines/inat_bioclip/inat_bioclip_seed${s}_clustering_ce${ce}.pt
    [ -f "$f" ] && $EV --emb_path "$f" --tag bioclip_head_s${s}_ce${ce} || echo "skip $f"
  done
done
module purge
