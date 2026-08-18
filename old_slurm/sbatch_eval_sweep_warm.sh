#!/bin/bash
#SBATCH --job-name=eval_sweep_warm
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_sweep_warm_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset cifar100"
for ROOT in False True; do
  for W in 0p1 0p5 1p0 2p0; do
    f=baselines/rebuild/c100_wsweep_w${W}_seed1_clustering_ce300.pt
    [ -f "$f" ] && $EV --emb_path "$f" --use_root ${ROOT} --tag wsweep_w${W}_s1_root${ROOT}
  done
  for W in 1p0 5p0; do
    f=baselines/rebuild/c100_warmstart_w${W}_seed1_clustering_ce300.pt
    [ -f "$f" ] && $EV --emb_path "$f" --use_root ${ROOT} --tag warmstart_w${W}_s1_root${ROOT}
  done
done
module purge
