#!/bin/bash
#SBATCH --job-name=eval_wideb
#SBATCH --time=3:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/eval_wideb_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# hDP for the wide-boundary CIFAR-100 d=16 model (HYP_MAX_ENORM_EPS=1e-4).
# Root-inclusive, matching Table 2. All four linkages: the whole point of the
# rerun is that the distance matrix is now 99.95% resolved, so a spread across
# linkages here is real signal rather than tie-breaking.
S=$((SLURM_ARRAY_TASK_ID + 1))
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/c100_d16_wideboundary/c100_d16_wideb_seed${S}_clustering_ce300.pt \
    --manifold poincare --c 1.0 --dataset cifar100 \
    --use_root True --all_linkages --tag wideb_c100d16_s${S}_ce300
module purge
