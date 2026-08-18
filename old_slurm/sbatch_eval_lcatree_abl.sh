#!/bin/bash
#SBATCH --job-name=eval_lcatree_abl
#SBATCH --time=6:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_lcatree_abl_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Retry of the LCA-tree half of 663068, which failed with an invalid CUDA
# device ordinal. No "|| skip" this time: a failure should stop the job.
LCA="uv run python scripts/eval_lca_tree.py --manifold poincare --c 1.0 --dataset cifar100"
for s in 1 2 3; do
  ${LCA} --emb_path baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce300.pt --tag lcatree_abl_joint_s${s}
  ${LCA} --emb_path baselines/rebuild/c100_conly_d16_seed${s}_clustering_ce300.pt   --tag lcatree_abl_conly_s${s}
  ${LCA} --emb_path baselines/rebuild/c100_lhonly_d16_seed${s}_clustering_ce300.pt  --tag lcatree_abl_lhonly_s${s}
done
module purge
