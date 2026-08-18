#!/bin/bash
#SBATCH --job-name=link_base
#SBATCH --time=8:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/link_base_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# The SAME linkage sweep (single/average/complete/Ward), run on the baselines.
#
# Purpose: echd's worry is that "the headline gains may correlate with the
# linkage step rather than with the learned hierarchy." The decisive evidence
# is not a defence of Ward -- it is showing that we beat the baselines under
# EVERY linkage. If the ordering is linkage-invariant, the linkage cannot be
# what produces the gap, and the argument is over without ever discussing
# which hDP variant anything is on.
#
# Each array task is one Table 2 row, and each row is scored on the SAME
# use_root convention as that row's published Ward number, so every cell we
# report sits next to a Ward value the reviewer can match against the paper:
#   arm 0  CIFAR-100 d=16      use_root=True   (paper 0.846)
#   arm 1  TinyImageNet d=16   use_root=False  (paper 0.728)
#   arm 2  TinyImageNet d=128  use_root=True   (paper 0.834, reproduced 0.836)
#
# Baseline manifolds/curvatures are copied from the evals that produced the
# numbers already in the paper: HIER is Poincare c=0.1, SCAN and SeCu are
# Euclidean. Changing either would make the comparison non-matched.
# ============================================================================

EV="uv run python scripts/eval_baseline_embeddings.py --all_linkages"

case $SLURM_ARRAY_TASK_ID in
  0)
    DS=cifar100; ROOT=True; NAME=c100d16
    ARMS=(
      "baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb16_eval_75.pt:poincare:0.1:hier_s1"
      "baselines/euclidean_npy_converted/scan_c100_test.pt:euclidean:1.0:scan"
      "baselines/euclidean_npy_converted/secu_c100d16_test.pt:euclidean:1.0:secu"
    )
    ;;
  1)
    DS=tiny_imagenet; ROOT=False; NAME=tid16
    ARMS=(
      "baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb16_lowdim_eval_95.pt:poincare:0.1:hier_s1"
      "baselines/euclidean_npy_converted/scan_ti_scan_ti_s42_test.pt:euclidean:1.0:scan"
      "baselines/euclidean_npy_converted/secu_ti_s42_test.pt:euclidean:1.0:secu"
    )
    ;;
  2)
    DS=tiny_imagenet; ROOT=True; NAME=tid128
    ARMS=(
      "baselines/HIER-CVPR23/logs/TinyImageNet/resnet50_seed1_emb128_eval_90.pt:poincare:0.1:hier_s1"
      "baselines/euclidean_npy_converted/scan_ti_scan_ti_s42_test.pt:euclidean:1.0:scan"
      "baselines/euclidean_npy_converted/secu_ti_s42_test.pt:euclidean:1.0:secu"
    )
    ;;
esac

for a in "${ARMS[@]}"; do
  IFS=':' read -r P MAN CURV TAG <<< "$a"
  if [ ! -f "$P" ]; then echo "absent $P"; continue; fi
  echo "=== linkbase_${NAME}_${TAG} (root=${ROOT}) ==="
  $EV --emb_path "$P" --manifold "$MAN" --c "$CURV" --dataset ${DS} \
      --use_root ${ROOT} --tag linkbase_${NAME}_${TAG} || echo "FAILED ${TAG}"
done

module purge
