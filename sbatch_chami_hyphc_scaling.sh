#!/bin/bash
#SBATCH --job-name=chami_scale
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/chami_hyphc_scaling_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-9

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# Chami HypHC scaling sweep — runtime curve + DP-at-N comparison.
#
# Reviewer asked for two things to back the "doesn't scale" claim:
#   (1) a runtime/memory curve as N grows
#   (2) DP comparison on small / sampled datasets (where Chami can run)
#
# This array sweeps N up to 25k on CIFAR-100 and TinyImageNet at d=2, the
# rank Chami et al. (NeurIPS 2020) use throughout their paper. Raw-pixel
# cosine similarity throughout — no SSL or DINO features anywhere; faithful
# to Chami et al. 25k is the cap: it already gives a 25× span over N=1k and
# the runtime/memory curve is monotone enough to extrapolate the cliff.
#
# Each job writes a sidecar JSON (success or OOM) to
#   baselines/chami_hyphc_scaling/{tag}.json
# Successful runs additionally write a HIER-format .pt that
# scripts/eval_baseline_embeddings.py can score for hDP.
#
# Array decoding (10 tasks total):
#   task = dataset_idx * 5 + n_idx
#   dataset_idx : 0=cifar100,    1=tiny_imagenet
#   n_idx       : 0..4 -> N in {1000, 2500, 5000, 10000, 25000}
#
# All 10 tasks run in parallel (no throttle) — get the full sweep back fast.
# ============================================================================

DATASETS=(cifar100 tiny_imagenet)
N_GRID=(1000 2500 5000 10000 25000)

DS_IDX=$((SLURM_ARRAY_TASK_ID / 5))
N_IDX=$((SLURM_ARRAY_TASK_ID % 5))

DIM=2
DATASET=${DATASETS[$DS_IDX]}
N=${N_GRID[$N_IDX]}
SEED=1

OUT_DIR=baselines/chami_hyphc_scaling
TAG=${DATASET}_n${N}_d${DIM}_s${SEED}

echo "=== task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} N=${N} dim=${DIM} seed=${SEED} ==="
date
echo "host=$(hostname)  job=$SLURM_JOB_ID"

# Triplet count is N^2 (--num_triples 0 = chami paper sampling), so per-epoch
# cost grows quadratically. 50 epochs is Chami's UCI default; we keep it fixed
# so wallclock vs N reflects the algorithm, not the schedule. Big N may hit the
# 12h walltime — that becomes a "didn't finish in 12h" data point on the curve,
# which is exactly the scaling story we're after.
EPOCHS=50

uv run python scripts/run_chami_hyphc_scaling.py \
    --dataset ${DATASET} \
    --n_samples ${N} \
    --rank ${DIM} \
    --seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size 256 \
    --num_triples 0 \
    --learning_rate 1e-3 \
    --temperature 0.05 \
    --anneal_every 20 \
    --anneal_factor 0.5 \
    --data_root data \
    --out_dir ${OUT_DIR} \
    --tag ${TAG}

# Score hDP on the embedding (only succeeds if .pt was written, i.e. status=ok).
# We let this fail silently for OOM runs — the sidecar will say status=oom_*
# and the aggregator picks that up.
PT_FILE=${OUT_DIR}/${TAG}.pt
if [ -f "${PT_FILE}" ]; then
    echo "=== eval hDP on ${PT_FILE} ==="
    uv run python scripts/eval_baseline_embeddings.py \
        --emb_path ${PT_FILE} \
        --manifold poincare --c 1.0 \
        --dataset ${DATASET} \
        --tag chami_scaling_${TAG} \
        --out_dir results_baseline_embeddings || echo "[eval] failed; skipping"
else
    echo "=== no .pt written for ${TAG}; status was OOM/error ==="
fi

date
module purge
