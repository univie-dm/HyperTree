#!/bin/bash
#SBATCH --job-name=lcatree_evals
#SBATCH --time=6:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/lcatree_evals_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
# hDP on the LCA-induced tree (Reviewer echd) — ours + HIER + random floor,
# full test sets, rectified LCA decoding.
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
EV="uv run python scripts/eval_lca_tree.py"
for s in 1 2 3; do
  for ce in 0 300; do
    $EV --emb_path baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce${ce}.pt \
        --manifold poincare --c 1.0 --dataset cifar100 --tag lcatree_ours_c100d16_s${s}_ce${ce}
  done
done
$EV --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb16_eval_75.pt --manifold poincare --c 0.1 --dataset cifar100 --tag lcatree_hier_c100d16_s1
$EV --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed2_emb16_eval_90.pt --manifold poincare --c 0.1 --dataset cifar100 --tag lcatree_hier_c100d16_s2
$EV --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed3_emb16_eval_95.pt --manifold poincare --c 0.1 --dataset cifar100 --tag lcatree_hier_c100d16_s3
$EV --emb_path baselines/euclidean_npy_converted/random_c100_poincare_d16_test.pt --manifold poincare --c 1.0 --dataset cifar100 --tag lcatree_random_c100
for s in 42 43 44; do
  $EV --emb_path baselines/ti_d128_full600/ti_d128_seed${s}_ce600.pt \
      --manifold poincare --c 1.0 --dataset tiny_imagenet --tag lcatree_ours_ti_d128_s${s}
done
$EV --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb128_eval_90.pt --manifold poincare --c 0.1 --dataset tiny_imagenet --tag lcatree_hier_ti_d128_s1
$EV --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed2_emb128_eval_80.pt --manifold poincare --c 0.1 --dataset tiny_imagenet --tag lcatree_hier_ti_d128_s2
$EV --emb_path baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed3_emb128_eval_95.pt --manifold poincare --c 0.1 --dataset tiny_imagenet --tag lcatree_hier_ti_d128_s3
