#!/bin/bash
#SBATCH --job-name=eval_traj_peak
#SBATCH --time=8:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_traj_peak_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# TRAJECTORY EVAL: joint vs contrastive-only, every saved checkpoint.
#
# Question: the joint arm is unstable (seed sd 0.182 at use_root=False vs 0.046
# for L_C-only, and hDP swings 0.37 -> 0.72 across adjacent epochs), so its
# endpoint may understate what it can reach. Does its PEAK beat L_C-only's?
#
# Evaluated SYMMETRICALLY: every checkpoint of both arms, so peak is compared
# against peak and endpoint against endpoint. Comparing the joint arm's best
# epoch against the contrastive arm's final epoch would be the same selection
# error the question is trying to avoid.
#
# CAVEAT for reporting: a maximum taken over the test set across epochs is a
# selection-on-test statistic, not a protocol. It is diagnostic (a high peak
# with an unstable trajectory points to over-weighting plus a missing stopping
# rule, rather than to the loss being inert), but it cannot be quoted as a
# headline number without a validation-based early-stopping criterion.
#
# Seed 1, both use_root variants.
# ============================================================================

EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset cifar100"
S=1

for ROOT in False True; do
  for f in baselines/c100_d16_full300/c100_d16_seed${S}_clustering_ce*.pt; do
    ce=$(basename "$f" .pt); ce=${ce##*_ce}
    $EV --emb_path "$f" --use_root ${ROOT} --tag traj_joint_s${S}_ce${ce}_root${ROOT}
  done
  for f in baselines/rebuild/c100_conly_d16_seed${S}_clustering_ce*.pt; do
    ce=$(basename "$f" .pt); ce=${ce##*_ce}
    $EV --emb_path "$f" --use_root ${ROOT} --tag traj_conly_s${S}_ce${ce}_root${ROOT}
  done
done

module purge
