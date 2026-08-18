#!/bin/bash
#SBATCH --job-name=eval_determ
#SBATCH --time=2:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_determ_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Same file, same settings, five times on one node. Quantifies how much of the
# reported hDP spread is the metric rather than the model. Two arms so we can
# tell whether the instability is arm-specific.
EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset cifar100 --use_root False"
for i in 1 2 3 4 5; do
  $EV --emb_path baselines/c100_d16_full300/c100_d16_seed1_clustering_ce300.pt --tag determ_joint_s1_r${i}
  $EV --emb_path baselines/rebuild/c100_conly_d16_seed1_clustering_ce300.pt   --tag determ_conly_s1_r${i}
done
module purge
