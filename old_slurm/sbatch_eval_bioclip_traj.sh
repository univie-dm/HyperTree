#!/bin/bash
#SBATCH --job-name=eval_bioclip_traj
#SBATCH --time=3:00:00
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -w shelob
#SBATCH --output=logs/eval_bioclip_traj_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
# Where does the BioCLIP degradation start? Free: the dumps already exist.
EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 0.3 --dataset inat18 --use_root False"
for ce in 10 20 30 40; do
  f=baselines/inat_bioclip/inat_bioclip_seed1_clustering_ce${ce}.pt
  [ -f "$f" ] && $EV --emb_path "$f" --tag bioclip_traj_s1_ce${ce}
done
module purge
