#!/bin/bash
#SBATCH --job-name=rb_ours_evals
#SBATCH --time=4:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/rebuild_ours_evals_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
EV="uv run python scripts/eval_baseline_embeddings.py --use_root False"
for arm in c100_hyp_d16 c100_poin_d32 c100_poin_d128; do
  case $arm in c100_hyp_d16) MAN="hyperboloid";; *) MAN="poincare";; esac
  for s in 1 2 3; do
    for ce in 0 290 300; do
      f=baselines/rebuild/${arm}_seed${s}_clustering_ce${ce}.pt
      [ -f "$f" ] && $EV --emb_path "$f" --dataset cifar100 --manifold $MAN --c 1.0 --tag rebuildC_${arm}_s${s}_ce${ce} || echo "skip $f"
    done
  done
done
for arm in ti_poin_d16 ti_poin_d128_simclr ti_hyp_d128_simclr; do
  case $arm in ti_hyp_*) MAN="hyperboloid";; *) MAN="poincare";; esac
  for s in 42 43 44; do
    for ce in 0 580 590 600; do
      f=baselines/rebuild/${arm}_seed${s}_clustering_ce${ce}.pt
      [ -f "$f" ] && $EV --emb_path "$f" --dataset tiny_imagenet --manifold $MAN --c 1.0 --tag rebuildC_${arm}_s${s}_ce${ce} || echo "skip $f"
    done
  done
done
