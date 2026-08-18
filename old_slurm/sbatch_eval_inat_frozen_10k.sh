#!/bin/bash
#SBATCH --job-name=inat_froz_10k
#SBATCH --time=3:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/inat_froz_10k_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNaturalist-2018 frozen-backbone controls on the 10k STRATIFIED SUBSAMPLE.
#
# Why: Gy9D Q1 puts two things next to each other that are not on the same
# protocol. The frozen-backbone table (random 0.138 / DINO 0.281 / BioCLIP
# 0.552) is the FULL validation split, while the BioCLIP head-training numbers
# quoted in the same answer (epoch-0 anchor 0.293 rising to 0.344 at epoch 10)
# come from the 10k class-stratified subsample the paper uses for iNat. A
# reader therefore sees frozen BioCLIP at 0.552 and its own epoch-0 anchor at
# 0.293 and reasonably asks why epoch 0 is worse than the raw features.
#
# Re-running the three frozen backbones on the 10k subsample puts the table on
# the same protocol as the 0.293 anchor, so the two can be read together and
# the explanatory clause is no longer needed.
#
# Protocol matched to the head-training evaluation: root excluded, 10k
# stratified subsample (same flags as the bioclip ablation runs). All three
# arms in one array on one node type, because cross-node hDP drift on this
# cluster reaches 0.19 and would swamp the comparison.
#
# Note: the 10k subsample is not perfectly deterministic across evaluations
# (~+-0.05-0.10), so treat small differences between arms with care.
# ============================================================================

EMB=(
  "baselines/frozen/inat18_val_random.pt:froz10k_inat_random"
  "baselines/frozen/inat18_val_dino_rn50.pt:froz10k_inat_dino_rn50"
  "baselines/frozen/inat18_val_bioclip.pt:froz10k_inat_bioclip"
)
IFS=':' read -r EMB_PATH TAG <<< "${EMB[$SLURM_ARRAY_TASK_ID]}"

echo "=== inat frozen 10k stratified: ${TAG} ==="

uv run python scripts/eval_baseline_embeddings.py \
    --emb_path "${EMB_PATH}" \
    --manifold euclidean --c 1.0 \
    --dataset inat18 \
    --use_root False \
    --test_sample_size 10000 \
    --tag "${TAG}"

module purge
