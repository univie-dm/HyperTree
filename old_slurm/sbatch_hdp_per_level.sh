#!/bin/bash
#SBATCH --job-name=hdp_perlevel
#SBATCH --time=10:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/hdp_perlevel_%A.log
#SBATCH -A datamining
#SBATCH -x vader

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# PER-LEVEL hDP (Reviewer Y8o7-Q1).
#
# The headline hDP pools every ancestor cluster into one number, so "recovers
# structure at every depth" and "only sharpens the coarsest split" score alike.
# This decomposes it by depth in the WordNet hierarchy, each level against its
# own random-embedding floor.
#
# Decision rule:
#   * gains spread across depths -> genuine multi-level structure; answers Q1.
#   * gains only at depth 1-2    -> Y8o7 is right, and we say so and scope the
#                                   claim rather than be caught on it.
#
# CIFAR-100 d=16, pinned protocol (use_root=False, Ward). Arms: ours epoch-0
# anchor, ours trained, L_H-only, and supervised HIER (c=0.1).
# ============================================================================

EV="uv run python scripts/eval_hdp_per_level.py --dataset cifar100 --linkage ward"

for s in 1 2 3; do
  echo "=== ours seed${s} ce0 (anchor) ==="
  ${EV} --emb_path baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce0.pt \
        --manifold poincare --c 1.0 --tag ours_c100d16_s${s}_ce0
  echo "=== ours seed${s} ce300 (trained) ==="
  ${EV} --emb_path baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce300.pt \
        --manifold poincare --c 1.0 --tag ours_c100d16_s${s}_ce300
done

echo "=== L_H-only seed1 ce300 ==="
${EV} --emb_path baselines/rebuild/c100_lhonly_d16_seed1_clustering_ce300.pt \
      --manifold poincare --c 1.0 --tag lhonly_c100d16_s1_ce300

echo "=== HIER (supervised, c=0.1) ==="
${EV} --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb16_eval_75.pt \
      --manifold poincare --c 0.1 --tag hier_c100d16_s1
${EV} --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed2_emb16_eval_90.pt \
      --manifold poincare --c 0.1 --tag hier_c100d16_s2
${EV} --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed3_emb16_eval_95.pt \
      --manifold poincare --c 0.1 --tag hier_c100d16_s3

module purge
