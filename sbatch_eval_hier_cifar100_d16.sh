#!/bin/bash
#SBATCH --job-name=eval_hier_d16
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_hier_cifar100_d16_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# Pipe HIER d=16 eval embeddings through our DP pipeline at every saved
# checkpoint epoch, for all 3 seeds. Skips missing files so it is safe to
# run before all training jobs finish.
for SEED in 1 2 3; do
    for EPOCH in 95 90 75 55 45 30 15 5; do
        EMB_PATH=baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed${SEED}_emb16_eval_${EPOCH}.pt
        if [ ! -f "$EMB_PATH" ]; then
            echo "skip seed=${SEED} ep=${EPOCH}: $EMB_PATH not found"
            continue
        fi
        uv run python scripts/eval_baseline_embeddings.py \
            --emb_path $EMB_PATH \
            --manifold poincare --c 0.1 --dataset cifar100 \
            --tag hier_c100_s${SEED}_d16_e${EPOCH}
    done
done

module purge
