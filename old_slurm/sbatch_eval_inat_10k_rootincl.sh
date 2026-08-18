#!/bin/bash
#SBATCH --job-name=inat_10k_root
#SBATCH --time=3:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/inat_10k_root_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-5

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNaturalist-2018 controls on TABLE 2's PROTOCOL: 10k class-stratified
# subsample, root INCLUDED.
#
# Why: Y8o7-Q2 and echd-Q2 want to read the frozen backbone against the paper's
# epoch-0 anchor (0.827) and trained result (0.872). Those two come from Table 2,
# which is 10k stratified and root-inclusive. The frozen controls currently in
# the rebuttal are root-inclusive but on the FULL 24,426-image split, so the
# comparison still crosses a protocol boundary on the split axis. Re-running the
# frozen rows here at 10k puts every number in that paragraph on one protocol.
#
# Arms 0-2: frozen features (random floor / DINO ResNet-50 / BioCLIP).
# Arms 3-5: the BioCLIP epoch-0 anchor, three seeds, from the dumps written by
#           the musica ablation. This gives BioCLIP a matched anchor of its own,
#           the counterpart to DINO's 0.827.
#
# All six arms in one array on one node type: cross-node hDP drift on this
# cluster reaches 0.19 and would swamp every difference being measured here.
# Note the 10k subsample is not perfectly deterministic across evaluations
# (~+-0.05-0.10 observed), so read small gaps with care.
# ============================================================================

ARMS=(
  "baselines/frozen/inat18_val_random.pt:r10k_inat_random"
  "baselines/frozen/inat18_val_dino_rn50.pt:r10k_inat_dino_rn50"
  "baselines/frozen/inat18_val_bioclip.pt:r10k_inat_bioclip"
  "baselines/inat_bioclip/inat_bioclip_seed1_clustering_ce0.pt:r10k_bioclip_ce0_s1"
  "baselines/inat_bioclip/inat_bioclip_seed2_clustering_ce0.pt:r10k_bioclip_ce0_s2"
  "baselines/inat_bioclip/inat_bioclip_seed3_clustering_ce0.pt:r10k_bioclip_ce0_s3"
)
IFS=':' read -r EMB_PATH TAG <<< "${ARMS[$SLURM_ARRAY_TASK_ID]}"

# The frozen arms are raw Euclidean features; the ce0 arms are Poincare
# embeddings from the head at initialisation and must use the manifold metric.
case $SLURM_ARRAY_TASK_ID in
  0|1|2) MANIFOLD="euclidean"; CURV=1.0 ;;
  *)     MANIFOLD="poincare";  CURV=0.3 ;;
esac

echo "=== inat 10k root-inclusive: ${TAG} (${MANIFOLD}) ==="

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path "${EMB_PATH}" \
    --manifold "${MANIFOLD}" --c ${CURV} \
    --dataset inat18 \
    --use_root True \
    --test_sample_size 10000 \
    --tag "${TAG}"

module purge
