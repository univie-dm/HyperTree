#!/bin/bash
#SBATCH --job-name=eval_ti_wideb
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/eval_ti_wideb_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Settles the TinyImageNet d=16 complete-linkage cell. Root-excluded, matching
# that row's published Ward number (0.728).
SEEDS=(42 43 44); S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/ti_d16_wideboundary/ti_d16_wideb_seed${S}_clustering_ce600.pt \
    --manifold poincare --c 1.0 --dataset tiny_imagenet \
    --use_root False --all_linkages --tag wideb_tid16_s${S}_ce600
module purge
