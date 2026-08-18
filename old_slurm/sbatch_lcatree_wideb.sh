#!/bin/bash
#SBATCH --job-name=lca_wideb
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/lca_wideb_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-5
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# LCA-induced-tree hDP on the WIDE-BOUNDARY CIFAR-100 checkpoints.
#
# compute_lca_depth() builds the geodesic-circle centre and then reads
# dist_0(lca) = 2*arctanh(||lca||); on the boundary-saturated original
# checkpoints (1-||x|| ~ 3e-8) that arctanh is evaluated where it has lost
# most of its precision, so the published LCA figures sit on the same
# degenerate path as the old linkage numbers. The wide-boundary model keeps
# 1-||x|| at 1e-4, where the existing code is numerically fine -- so this
# measures the LCA readout without writing a second implementation of it.
#
# ce0 anchors included so the trained-vs-untrained comparison is matched.
SEEDS=(1 2 3)
case $SLURM_ARRAY_TASK_ID in
  0|1|2) S=${SEEDS[$SLURM_ARRAY_TASK_ID]}; CE=300 ;;
  3|4|5) S=${SEEDS[$((SLURM_ARRAY_TASK_ID-3))]}; CE=0 ;;
esac
P=baselines/c100_d16_wideboundary/c100_d16_wideb_seed${S}_clustering_ce${CE}.pt
if [ ! -f "$P" ]; then echo "absent $P"; exit 0; fi

uv run python scripts/eval_lca_tree.py \
    --emb_path "$P" --manifold poincare --c 1.0 --dataset cifar100 \
    --lca_version ours --tag lcatree_wideb_c100d16_s${S}_ce${CE}
module purge
