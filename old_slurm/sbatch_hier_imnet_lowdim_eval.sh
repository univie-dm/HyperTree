#!/bin/bash
#SBATCH --job-name=hier_imnet_lowdim_eval
#SBATCH --time=02:00:00
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --output=logs/hier_imnet_lowdim_eval_%j.log
#SBATCH -A datamining
#SBATCH --exclude=vader,galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

cd /srv/home/users/sidakk95cs/hyperbolic-clustering

for d in 16 32 64; do
  for s in 1 2 3; do
    case "${s}_${d}" in
      "2_32") EP=3 ;;
      "3_64") EP=9 ;;
      *) EP=12 ;;
    esac
    EMBPATH="baselines/HIER-CVPR23/logs/ImageNet/resnet50_seed${s}_emb${d}_eval_${EP}.pt"
    TAG="hier_imnet_s${s}_d${d}_e${EP}"
    OUT="results_baseline_embeddings/${TAG}.json"
    if [ -f "$OUT" ]; then echo "[skip] $TAG (exists)"; continue; fi
    echo "=== eval $TAG ==="
    uv run python scripts/eval_baseline_embeddings.py \
      --emb_path "$EMBPATH" \
      --manifold poincare --c 0.1 \
      --dataset imagenet \
      --tag "$TAG" \
      --use_root False \
      --test_sample_size 10000
  done
done

module purge
