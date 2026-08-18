#!/bin/bash
#SBATCH --job-name=eval_c100_root_false
#SBATCH --time=0:15:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -w galadriel
#SBATCH --output=logs/eval_c100_root_false_%A.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

set -e

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# Re-eval HIER CIFAR-100 d=16 seed=1 at e=75 with dendrogrampurity_use_root=False
# to check how much the absolute hDP_ward shifts under the looser convention.

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb16_eval_75.pt \
    --manifold poincare --c 0.1 --dataset cifar100 \
    --use_root False \
    --tag hier_c100_s1_d16_e75_root_false

module purge
