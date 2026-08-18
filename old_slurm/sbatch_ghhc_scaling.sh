#!/bin/bash
#SBATCH --job-name=ghhc_scale
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/ghhc_scaling_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-9

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# Monath et al.'s gHHC (KDD 2019) scaling sweep — runtime curve + DP-at-N.
#
# Companion to sbatch_chami_hyphc_scaling.sh: we cap N at 25k for both
# datasets at d=2 (gHHC's Poincaré disk default). Raw-pixel cosine via
# gHHC's own zero-mean + unit-norm + max_norm clip preprocessing; init via
# `randhac` (gHHC's recommended for non-ImageNet datasets).
#
# Each job writes a sidecar JSON in baselines/ghhc_scaling/{tag}.json with
# status ∈ {ok, oom_host, oom_gpu, error} per phase. Successful runs also
# save a HIER-format .pt that scripts/eval_baseline_embeddings.py can score.
#
# Array decoding (10 tasks total):
#   task = dataset_idx * 5 + n_idx
#   dataset_idx : 0=cifar100,    1=tiny_imagenet
#   n_idx       : 0..4 -> N in {1000, 2500, 5000, 10000, 25000}
#
# All 10 tasks run in parallel (no throttle).
# ============================================================================

DATASETS=(cifar100 tiny_imagenet)
N_GRID=(1000 2500 5000 10000 25000)

DS_IDX=$((SLURM_ARRAY_TASK_ID / 5))
N_IDX=$((SLURM_ARRAY_TASK_ID % 5))

DATASET=${DATASETS[$DS_IDX]}
N=${N_GRID[$N_IDX]}
SEED=1

OUT_DIR=baselines/ghhc_scaling
TAG=${DATASET}_n${N}_s${SEED}

echo "=== task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} N=${N} seed=${SEED} ==="
date
echo "host=$(hostname)  job=$SLURM_JOB_ID"

# gHHC defaults from configs/glass/glass.json: num_iterations=10 (default),
# batch_size=500, num_samples=50000. We keep the 50k triplet budget fixed
# across N (matches Monath's KDD'19 setup), so wallclock vs N reflects what
# scales: kNN build, training compute on a fixed budget, and assignment.
uv run python scripts/run_ghhc_scaling.py \
    --dataset ${DATASET} \
    --n_samples ${N} \
    --seed ${SEED} \
    --num_triples 50000 \
    --num_iterations 10 \
    --batch_size 500 \
    --episode_size 5000 \
    --learning_rate 0.01 \
    --init_method randhac \
    --data_root data \
    --out_dir ${OUT_DIR} \
    --tag ${TAG}

PT_FILE=${OUT_DIR}/${TAG}.pt
if [ -f "${PT_FILE}" ]; then
    echo "=== eval hDP on ${PT_FILE} ==="
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path ${PT_FILE} \
        --manifold poincare --c 1.0 \
        --dataset ${DATASET} \
        --tag ghhc_scaling_${TAG} \
        --out_dir results_baseline_embeddings || echo "[eval] failed; skipping"
else
    echo "=== no .pt written for ${TAG}; status was OOM/error ==="
fi

date
module purge
