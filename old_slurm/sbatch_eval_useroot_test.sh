#!/bin/bash
#SBATCH --job-name=eval_useroot_test
#SBATCH --time=0:30:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_useroot_test_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Hypothesis: use_root=False reproduces the May-1 discriminative hDP (0.483)
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb128_eval_90.pt \
    --manifold poincare --c 0.1 --dataset tiny_imagenet \
    --use_root False \
    --tag hier_ti_s1_d128_e90_norootflag
# And the random TI floor without root, for the discriminativeness check
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path baselines/euclidean_npy_converted/random_ti_test.pt \
    --manifold euclidean --dataset tiny_imagenet \
    --use_root False \
    --tag random_ti_nulltest_noroot
module purge
