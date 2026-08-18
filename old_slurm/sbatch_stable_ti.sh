#!/bin/bash
#SBATCH --job-name=stable_ti
#SBATCH --time=6:00:00
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/stable_ti_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-5
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# TinyImageNet is MORE boundary-degenerate than CIFAR-100 (0.12% and 0.39%
# distinct pairwise distances at d=16 and d=128, against CIFAR-100's 1.6%), so
# its rows in the linkage table need the stable-distance treatment too.
# Conventions match each row's published Ward number: d=16 root-excluded
# (paper 0.728), d=128 root-inclusive (paper 0.834).
SEEDS=(42 43 44)
case $SLURM_ARRAY_TASK_ID in
  0|1|2) S=${SEEDS[$SLURM_ARRAY_TASK_ID]}
         P=baselines/rebuild/ti_poin_d16_seed${S}_clustering_ce600.pt
         ROOT=False; TAG=stable_tid16_s${S}_ce600 ;;
  3|4|5) S=${SEEDS[$((SLURM_ARRAY_TASK_ID-3))]}
         P=baselines/ti_d128_full600/ti_d128_seed${S}_ce600.pt
         ROOT=True; TAG=stable_tid128_s${S}_ce600 ;;
esac

uv run python scripts/eval_stable_distance.py \
    --emb_path "$P" --dataset tiny_imagenet --c 1.0 \
    --use_root ${ROOT} --tag ${TAG}
module purge
