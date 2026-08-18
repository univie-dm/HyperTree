#!/bin/bash
#SBATCH --job-name=chami_c100_60k
#SBATCH --time=16:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/chami_hyphc_cifar100_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-8%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# Chami HypHC (NeurIPS 2020) baseline on CIFAR-100 val.
#
# Why this is needed for the paper: §4.2.3 rectifies Chami's LCA flaw, but
# without an actual Chami baseline run there's no measured comparison to
# motivate the section. The `lca_version=chami` ablation only swaps the
# LCA — it's not Chami's full method.
#
# Chami's method:
#   - Per-leaf embedding parameters (one row per CIFAR-100 val image, n=10k)
#   - HypHC triplet loss with temperature annealing
#   - Cosine similarity on RAW images (paper.md:165 — this is the O(n^2)
#     bottleneck that makes their method intractable on larger datasets)
#
# 3 ranks × 3 seeds = 9 runs. Each takes ~30 min on 1×H100 at 50 epochs.
# After this, eval all 9 .pt embeddings via eval_baseline_embeddings.py to
# get hDP_ward / DP_ward / NMI for the §5.4 deep clustering table.
#
# Array decoding:
#   task = rank_idx * 3 + seed_idx
#   rank_idx: 0=16, 1=32, 2=128
#   seed_idx: 0=1, 1=2, 2=3
# ============================================================================

RANKS=(16 32 128)
SEEDS=(1 2 3)

RANK_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
RANK=${RANKS[$RANK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

OUT_PT="baselines/chami_hyphc/cifar100_all60k_d${RANK}_s${SEED}.pt"

echo "=== chami_c100 task ${SLURM_ARRAY_TASK_ID}: rank=${RANK} seed=${SEED} ==="

uv run python scripts/run_chami_hyphc_cifar100.py \
    --rank ${RANK} \
    --seed ${SEED} \
    --epochs 50 \
    --batch_size 256 \
    --learning_rate 1e-3 \
    --temperature 0.05 \
    --anneal_every 20 \
    --anneal_factor 0.5 \
    --num_samples 2000000 \
    --data_root data \
    --split all \
    --out_pt ${OUT_PT}
# NOTE: at n=60k, all-triples enumeration is C(60000,3) ~= 3.6e13 (the O(n^3)
# wall from Sec 3 — intractable), so HypHC is run with a sampled budget of
# 2M triples/epoch. This is itself ~74x larger than our method's ~27k
# triples/step sample-complexity bound (Thm 4.4), which is n-independent.

echo "=== eval ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${OUT_PT} \
    --manifold poincare --c 1.0 --use_root False \
    --dataset cifar100 \
    --test_sample_size 10000 \
    --tag chami_hyphc_cifar100_60k_d${RANK}_s${SEED}

module purge
