#!/bin/bash
#SBATCH --job-name=rb_evals
#SBATCH --time=8:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/rebuild_evals_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
# Pinned-protocol (variant C: use_root=False, native distances, 10k stratified
# for ImageNet) evals of all cached baseline embeddings, for the rebuttal
# number rebuild.
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
while IFS= read -r cmd; do
  echo ">>> $cmd"
  eval "$cmd" || echo "FAILED: $cmd"
done < rebuild_eval_cmds.txt
