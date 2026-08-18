#!/bin/bash
#SBATCH --job-name=pilotC_final300
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/pilotC_final300_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
# True-final (ce290/295/300) evals of the 300-epoch C100 d16 rerun under the
# pinned variant-C protocol (use_root=False, native poincare distance).
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
for s in 1 2 3; do
  for ce in 290 295 300; do
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce${ce}.pt \
        --dataset cifar100 --use_root False --manifold poincare --c 1.0 \
        --tag pilotC300_s${s}_ce${ce}
  done
done
module purge
