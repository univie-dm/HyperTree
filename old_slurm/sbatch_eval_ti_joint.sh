#!/bin/bash
#SBATCH --job-name=eval_ti_joint
#SBATCH --time=4:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/eval_ti_joint_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Re-evaluate the TI JOINT arm and the random floor in ONE job with the two
# ablation arms, so the whole triad is measured on a single node. The 0.724
# joint figure came from a different job, and cross-node float differences move
# Ward-hDP by up to 0.19 via linkage tie-breaking.
EVW="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --dataset tiny_imagenet --use_root False"
for s in 42 43 44; do
  $EVW --emb_path baselines/rebuild/ti_poin_d16_seed${s}_clustering_ce600.pt --tag ti_joint_s${s}_ward_rerun
  $EVW --emb_path baselines/rebuild/ti_conly_d16_seed${s}_clustering_ce600.pt  --tag ti_conly_s${s}_ward_rerun
  $EVW --emb_path baselines/rebuild/ti_lhonly_d16_seed${s}_clustering_ce600.pt --tag ti_lhonly_s${s}_ward_rerun
done
$EVW --emb_path baselines/euclidean_npy_converted/random_ti_test.pt --tag ti_floor_ward_rerun
module purge
