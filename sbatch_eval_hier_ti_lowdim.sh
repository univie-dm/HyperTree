#!/bin/bash
#SBATCH --job-name=eval_hier_ti_lowdim
#SBATCH --time=1:00:00
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_hier_ti_lowdim_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-1

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Eval HIER TinyImageNet d=16 / d=32 lowdim checkpoints (single seed each)
# through our matched pipeline: dendrogram purity + clustering metrics + n@k
# + R@k. HIER c=0.1.

DIMS=(16 32)
EPOCHS=(95 70)  # best checkpoint per dim
DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
EP=${EPOCHS[$SLURM_ARRAY_TASK_ID]}

EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb${DIM}_lowdim_eval_${EP}.pt

echo "=== eval HIER TI dim=${DIM} ep=${EP} ==="

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${EMB} \
    --manifold poincare \
    --c 0.1 \
    --dataset tiny_imagenet \
    --tag hier_ti_d${DIM}_lowdim_s1_ep${EP} \
    --use_root False

module purge
