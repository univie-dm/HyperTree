#!/bin/bash
#SBATCH --job-name=linkage_abl
#SBATCH --time=8:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/linkage_abl_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# hDP under single / average / complete / Ward linkage.
#
# Reviewer echd asks for the full linkage sweep on every dataset. Complete and
# Ward are already logged everywhere; single and average were never enabled,
# so this re-scores the saved embeddings with --all_linkages.
#
# Protocol is Table 2's: root INCLUDED, full split, Poincare geodesic distance.
# Each array task is one dataset/dimensionality, so the three trained seeds and
# the matched ce=0 anchor land on the same node -- cross-node hDP drift on this
# cluster reaches 0.19 and would swamp the between-linkage differences.
#
# Arm 0: CIFAR-100 d=16      (paper 0.846 Ward)
# Arm 1: TinyImageNet d=16   (paper 0.728 Ward)
# Arm 2: TinyImageNet d=128  (paper 0.834 Ward)
#
# ImageNet-1k and iNaturalist-2018 are absent because no trained embedding dump
# was kept for those runs -- only the L_H-only ImageNet ablation and the BioCLIP
# iNat ablation were dumped. Their linkage numbers can only come from wandb, and
# only for the linkages that were enabled at training time (complete + Ward).
# ============================================================================

EV="uv run python scripts/eval_baseline_embeddings.py --manifold poincare --c 1.0 --use_root True --all_linkages"

case $SLURM_ARRAY_TASK_ID in
  0)
    DS=cifar100; PREFIX=baselines/c100_d16_full300/c100_d16_seed; SUF=_clustering
    SEEDS="1 2 3"; FINAL=300; NAME=c100d16
    ;;
  1)
    DS=tiny_imagenet; PREFIX=baselines/rebuild/ti_poin_d16_seed; SUF=_clustering
    SEEDS="42 43 44"; FINAL=600; NAME=tid16
    ;;
  2)
    DS=tiny_imagenet; PREFIX=baselines/ti_d128_full600/ti_d128_seed; SUF=
    SEEDS="42 43 44"; FINAL=600; NAME=tid128
    ;;
esac

for s in $SEEDS; do
  for ce in 0 ${FINAL}; do
    f="${PREFIX}${s}${SUF}_ce${ce}.pt"
    if [ -f "$f" ]; then
      echo "=== link_${NAME}_s${s}_ce${ce} ==="
      $EV --dataset ${DS} --emb_path "$f" --tag link_${NAME}_s${s}_ce${ce}
    else
      echo "absent $f"
    fi
  done
done

# Matched random floor: Gaussian noise at the same dimensionality and sample
# count as the trained embeddings, scored through the identical path. Without
# it the linkage numbers cannot be read -- a linkage that scores 0.74 on a
# trained model and 0.74 on noise has told us nothing.
FIRST=$(echo $SEEDS | cut -d' ' -f1)
python - <<PY
import torch
x, y, i = torch.load("${PREFIX}${FIRST}${SUF}_ce${FINAL}.pt", weights_only=False)
g = torch.Generator().manual_seed(0)
r = torch.randn(x.shape[0], x.shape[1], generator=g) * 0.01
torch.save((r, y, i), "/tmp/link_floor_${NAME}.pt")
print("floor embedding", tuple(r.shape))
PY
echo "=== link_${NAME}_floor ==="
$EV --dataset ${DS} --emb_path "/tmp/link_floor_${NAME}.pt" --tag link_${NAME}_floor

module purge
