#!/bin/bash
#SBATCH --job-name=eval_ti_abl
#SBATCH --time=4:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_ti_abl_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# TI loss ablation, pinned to one node. Ward-hDP plus the LCA-induced tree,
# since the CIFAR result shows the two metrics disagree by design.
EVW="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset tiny_imagenet --use_root False"
LCA="uv run python scripts/eval_lca_tree.py --manifold poincare --c 1.0 --dataset tiny_imagenet"
for arm in conly lhonly; do
  for s in 42 43 44; do
    f=baselines/rebuild/ti_${arm}_d16_seed${s}_clustering_ce600.pt
    [ -f "$f" ] && $EVW --emb_path "$f" --tag ti_${arm}_s${s}_ward || echo "absent $f"
    [ -f "$f" ] && $LCA --emb_path "$f" --tag ti_${arm}_s${s}_lcatree || true
  done
done
module purge
