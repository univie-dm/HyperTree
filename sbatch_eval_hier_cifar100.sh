#!/bin/bash
#SBATCH --job-name=eval_hier
#SBATCH --time=01:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_hier_cifar100_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

for EPOCH in 95 90 75 55 45; do
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb128_eval_${EPOCH}.pt \
        --manifold poincare --c 0.1 --dataset cifar100 \
        --tag hier_c100_s1_d128_e${EPOCH}
done

module purge
