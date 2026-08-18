#!/bin/bash
#SBATCH --job-name=stable_c100
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/stable_c100_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# Does the CIFAR-100 headline number survive a numerically stable distance?
# Root-inclusive, matching Table 2's 0.846.
S=$((SLURM_ARRAY_TASK_ID + 1))
uv run python scripts/eval_stable_distance.py \
    --emb_path baselines/c100_d16_full300/c100_d16_seed${S}_clustering_ce300.pt \
    --dataset cifar100 --c 1.0 --use_root True \
    --tag stable_c100d16_s${S}_ce300
module purge
