#!/bin/bash
#SBATCH --job-name=chami_conv
#SBATCH --time=20:00:00
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -x vader,galadriel
#SBATCH --output=logs/chami_conv_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-1%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
set -e

# ============================================================================
# HypHC CONVERGENCE CONTROLS (rebuttal echd-Q4).
#
# The headline HypHC baseline (job 661974) ran 50 epochs on all n=60,000
# CIFAR-100 images and reached hDP 0.288 against a 0.252 random floor. Its
# training loss is flat to four decimals within each temperature regime:
#
#     ep 1  1.8025 | ep 3-16  1.8010 | ep 21-36  1.7988 | ep 41-50  1.7947
#
# i.e. 0.4% total movement, with the only changes coming from the temperature
# anneals at epochs 20 and 40. That is consistent with convergence, but also
# with the optimizer barely moving, and the two are not distinguishable from
# this curve alone. A reviewer can reasonably ask whether the baseline was
# simply undertrained.
#
# These two controls close that off:
#   0: 200 epochs at the published lr  -> is the plateau a truncation artefact?
#   1: 50 epochs at lr 1e-2 (10x)      -> is the plateau a step-size artefact?
#
# If both land at ~0.288, the baseline is converged and the comparison in
# echd-Q4 is fair. d=16 matches our headline dimensionality; 1 seed each.
#
# -x vader: vader runs ~40% slower than shelob on this workload.
# ============================================================================

CONFIGS=("200:1e-3:ep200" "50:1e-2:lr1e2")
IFS=':' read -r EPOCHS LR TAG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
RANK=16
SEED=1

OUT_PT="baselines/chami_hyphc/cifar100_all60k_d${RANK}_s${SEED}_${TAG}.pt"
echo "=== chami_conv ${TAG}: epochs=${EPOCHS} lr=${LR} rank=${RANK} seed=${SEED} ==="

uv run python scripts/run_chami_hyphc_cifar100.py \
    --rank ${RANK} \
    --seed ${SEED} \
    --epochs ${EPOCHS} \
    --batch_size 256 \
    --learning_rate ${LR} \
    --temperature 0.05 \
    --anneal_every 20 \
    --anneal_factor 0.5 \
    --num_samples 2000000 \
    --data_root data \
    --split all \
    --out_pt ${OUT_PT}

echo "=== eval (pinned protocol) ==="
uv run python scripts/eval_baseline_embeddings.py \
    --emb_path ${OUT_PT} \
    --manifold poincare --c 1.0 --use_root False \
    --dataset cifar100 \
    --test_sample_size 10000 \
    --tag chami_conv_${TAG}_d${RANK}_s${SEED}

module purge
