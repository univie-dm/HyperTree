#!/bin/bash
#SBATCH --job-name=hrc_reeval
#SBATCH --time=0:45:00
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH --output=logs/hrc_reeval_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-53%8

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

set -e
cd /srv/home/users/sidakk95cs/hyperbolic-clustering

# ============================================================================
# HRC re-eval of cached HIER baseline embeddings to add the new HRC metrics
# (hrc_sibling_triplet, hrc_origin_depth_*) to existing baseline JSONs.
# CPU-only — distance matrices are at most 10k x 10k (~400MB), no GPU needed.
#
# Manifest format (TSV): tag<TAB>dataset<TAB>manifold<TAB>c<TAB>emb_path
# Each task processes one row. Output JSON overwrites the existing one.
# ============================================================================

MANIFEST=baselines/hrc_reeval/manifest.tsv
LINE=$((SLURM_ARRAY_TASK_ID + 1))
ROW=$(sed -n "${LINE}p" "$MANIFEST")

TAG=$(echo "$ROW" | cut -f1)
DATASET=$(echo "$ROW" | cut -f2)
MANIFOLD=$(echo "$ROW" | cut -f3)
C=$(echo "$ROW" | cut -f4)
EMB=$(echo "$ROW" | cut -f5)

case "$DATASET" in
  cifar100) USE_ROOT=True ;;
  *)        USE_ROOT=False ;;
esac

echo "=== hrc_reeval task ${SLURM_ARRAY_TASK_ID}: tag=${TAG} ds=${DATASET} mf=${MANIFOLD} c=${C} use_root=${USE_ROOT} ==="
echo "    emb=${EMB}"

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path "${EMB}" \
    --manifold "${MANIFOLD}" \
    --c "${C}" \
    --dataset "${DATASET}" \
    --tag "${TAG}" \
    --use_root "${USE_ROOT}"

module purge
