#!/bin/bash
#SBATCH --job-name=lca_agg
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/lca_agg_%j.log
#SBATCH -A datamining

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Aggregate LCA wrong-rate trajectories from baselines/lca_wrong_rate/*.pt
# (CIFAR-100 d=16 + TinyImageNet d=128, 3 seeds each).
# Runs on GPU because TI d=128 has 50M pairs × 30 checkpoints × 3 seeds.

uv run python -m scripts.aggregate_lca_wrong_rate --c 1.0

module purge
