#!/bin/bash
#SBATCH --job-name=eval_abl_root
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_abl_root_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# Score the loss-ablation triad on BOTH hDP variants.
#
# The paper's headline CIFAR-100 d=16 figure (0.846) is the use_root=True
# variant; the pinned protocol used throughout the rebuild is use_root=False,
# where the same embeddings give 0.547. The L_C-only arm has so far only been
# scored on use_root=False (0.721). Comparing 0.721 against 0.846 would cross
# metrics, so this scores every arm on both variants.
#
# All three arms share an identical ce=0 anchor (0.2977 on seed 1 under
# use_root=False), i.e. the same encoder and architecture -- the comparison
# is matched.
# ============================================================================

EVB="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset cifar100"

for ROOT in True False; do
  for s in 1 2 3; do
    # joint (paper's arm)
    f=baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce300.pt
    [ -f "$f" ] && $EVB --emb_path "$f" --use_root ${ROOT} \
        --tag abl_joint_s${s}_ce300_root${ROOT} || echo "skip $f"
    # contrastive-only
    f=baselines/rebuild/c100_conly_d16_seed${s}_clustering_ce300.pt
    [ -f "$f" ] && $EVB --emb_path "$f" --use_root ${ROOT} \
        --tag abl_conly_s${s}_ce300_root${ROOT} || echo "skip $f"
    # L_H-only
    f=baselines/rebuild/c100_lhonly_d16_seed${s}_clustering_ce300.pt
    [ -f "$f" ] && $EVB --emb_path "$f" --use_root ${ROOT} \
        --tag abl_lhonly_s${s}_ce300_root${ROOT} || echo "skip $f"
    # shared anchor
    f=baselines/c100_d16_full300/c100_d16_seed${s}_clustering_ce0.pt
    [ -f "$f" ] && $EVB --emb_path "$f" --use_root ${ROOT} \
        --tag abl_anchor_s${s}_ce0_root${ROOT} || echo "skip $f"
  done
done

module purge
