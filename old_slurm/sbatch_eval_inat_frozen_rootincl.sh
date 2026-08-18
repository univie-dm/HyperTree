#!/bin/bash
#SBATCH --job-name=inat_froz_root
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/inat_froz_root_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNaturalist-2018 frozen-backbone controls under ROOT-INCLUSIVE hDP.
#
# The rebuttal's frozen controls (random 0.138 / DINO 0.281 / BioCLIP 0.552)
# were produced with the root cluster excluded, while the paper's Table 2 iNat
# entries (epoch-0 anchor 0.827, trained 0.872) are root-inclusive. Re-running
# the same three embedding files with the root included puts every number on
# the paper's convention.
#
# The question this answers: what does a RANDOM embedding score on iNat under
# root inclusion? On CIFAR-100 root inclusion lifts the random floor from 0.252
# to 0.721, so the iNat floor may also be high. If the root-inclusive random
# floor lands near frozen DINO's 0.810, the "backbone sits only modestly above
# the floor" reading survives on the matched convention and the frozen-control
# table can be restated against Table 2 directly. If the floor is low, frozen
# DINO at 0.810 against a 0.827 anchor says the hierarchy is largely already in
# the pretrained features, and the controls should stay on their own protocol.
#
# Same script, same split, same everything as the root-excluded run in
# sbatch_frozen_backbone_inat.sh -- only --use_root differs, so the pairs are
# directly comparable. All three arms run in one job array on one node type to
# avoid the cross-node hDP drift.
# ============================================================================

EMB=(
  "baselines/frozen/inat18_val_random.pt:rootIncl_inat_random"
  "baselines/frozen/inat18_val_dino_rn50.pt:rootIncl_inat_dino_rn50"
  "baselines/frozen/inat18_val_bioclip.pt:rootIncl_inat_bioclip"
)
IFS=':' read -r EMB_PATH TAG <<< "${EMB[$SLURM_ARRAY_TASK_ID]}"

echo "=== inat frozen root-inclusive: ${TAG} ==="

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path "${EMB_PATH}" \
    --manifold euclidean --c 1.0 \
    --dataset inat18 \
    --use_root True \
    --tag "${TAG}"

module purge
