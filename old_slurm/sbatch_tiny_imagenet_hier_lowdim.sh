#!/bin/bash
#SBATCH --job-name=hier_ti_lowdim
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH -w galadriel
#SBATCH --output=logs/hier_ti_lowdim_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-1%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# HIER on TinyImageNet at d=16 / d=32 — low-dim follow-up to the d=128
# multi-seed sbatch. 1 seed each (seed=1) to test whether HIER's d=128
# advantage holds at lower dim. On CIFAR-100, HIER's hDP gently increases
# as dim shrinks (d=128 0.813 → d=16 0.845); at TI d=128 HIER beats us
# by ~0.13 (estimated 0.848 vs ours 0.714). If HIER d=16/d=32 land near
# 0.85 the gap is dim-stable; if they drop toward our 0.71-0.73 range
# the low-dim regime becomes the place where ours competes.
#
# task_id 0 -> d=16
# task_id 1 -> d=32
# ============================================================================

cd /srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/HIER-CVPR23

mkdir -p logs/TinyImageNet

DIMS=(16 32)
DIM=${DIMS[$SLURM_ARRAY_TASK_ID]}
SEED=1

echo "=== hier_ti_lowdim task ${SLURM_ARRAY_TASK_ID}: dim=${DIM} seed=${SEED} ==="

uv run torchrun --standalone --nproc_per_node=1 hier/train.py \
    --data_path /srv/home/users/sidakk95cs/hyperbolic-clustering/data \
    --output_dir ./logs \
    --run_name seed${SEED}_emb${DIM}_lowdim \
    --seed ${SEED} \
    --lr 1e-4 --epochs 100 --batch_size 90 \
    --num_hproxies 512 --clip_r 2.3 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --lambda1 1 --lambda2 0.5 --topk 20 --mrg 0.1 \
    --dataset TinyImageNet --model resnet50 \
    --IPC 2 --loss PA --emb ${DIM} \
    --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 5 --lr_decay_gamma 0.5 \
    --save_emb True

module purge
