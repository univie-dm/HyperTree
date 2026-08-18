#!/bin/bash
#SBATCH --job-name=pilotC_geometry
#SBATCH --time=1:30:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/pilotC_geometry_%A.log
#SBATCH -A datamining
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge

# Pilot C: single implementation (our script), use_root=False, NATIVE distance
# per method (poincare for ours/HIER, euclidean for SCAN/SeCu/random).
# The geometry-respecting counterpart of pilot A.

set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
EV="uv run python scripts/eval_baseline_embeddings.py --dataset cifar100 --use_root False"

for s in 1 2 3; do
  for ce in 0 50; do
    $EV --emb_path baselines/lca_wrong_rate/c100_d16_seed${s}_ce${ce}.pt \
        --manifold poincare --c 1.0 --tag pilotC_ours_c100d16_s${s}_ce${ce}
  done
done
$EV --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed1_emb16_eval_75.pt --manifold poincare --c 0.1 --tag pilotC_hier_c100d16_s1
$EV --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed2_emb16_eval_90.pt --manifold poincare --c 0.1 --tag pilotC_hier_c100d16_s2
$EV --emb_path baselines/HIER-CVPR23/logs/CIFAR100/resnet50_seed3_emb16_eval_95.pt --manifold poincare --c 0.1 --tag pilotC_hier_c100d16_s3
$EV --emb_path baselines/euclidean_npy_converted/random_c100_test.pt --manifold euclidean --tag pilotC_random_c100
$EV --emb_path baselines/euclidean_npy_converted/secu_c100d16_test.pt --manifold euclidean --tag pilotC_secu_c100d16
$EV --emb_path baselines/euclidean_npy_converted/scan_c100_test.pt --manifold euclidean --tag pilotC_scan_c100

EVT="uv run python scripts/eval_baseline_embeddings.py --dataset tiny_imagenet --use_root False"
$EVT --emb_path baselines/lca_wrong_rate/ti_d128_seed42_ce0.pt --manifold poincare --c 1.0 --tag pilotC_ours_ti_d128_s42_ce0
$EVT --emb_path baselines/lca_wrong_rate/ti_d128_seed42_ce240.pt --manifold poincare --c 1.0 --tag pilotC_ours_ti_d128_s42_ce240
module purge
