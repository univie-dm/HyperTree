#!/bin/bash
#SBATCH --job-name=hier_ti
#SBATCH --time=20:00:00
#SBATCH -c 16
#SBATCH --mem=164G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/hier_ti_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-2%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# HIER on TinyImageNet as a supervised DML baseline for the paper.
#
# Protocol: full-class split (all 200 classes in train and eval) — same setup
# as our unsupervised clustering eval (100k train + 10k val).
#
# Hyperparams mirror the CIFAR-100 sbatch (same recipe family). Output
# embeddings are saved to baselines/HIER-CVPR23/logs/TinyImageNet/ and can
# be piped through scripts/eval_baseline_embeddings.py for hDP_ward / NMI /
# n@1 against the standard pipeline.
#
# 3-seed array (seeds 1, 2, 3) at d=128 (matches our TI d=128 main row).
# d=16/d=32 to follow once d=128 lands.
# ============================================================================

cd /srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/HIER-CVPR23

mkdir -p logs/TinyImageNet

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

uv run torchrun --standalone --nproc_per_node=1 hier/train.py \
    --data_path /srv/home/users/sidakk95cs/hyperbolic-clustering/data \
    --output_dir ./logs \
    --run_name seed${SEED}_emb128 \
    --seed ${SEED} \
    --lr 1e-4 --epochs 100 --batch_size 90 \
    --num_hproxies 512 --clip_r 2.3 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --lambda1 1 --lambda2 0.5 --topk 20 --mrg 0.1 \
    --dataset TinyImageNet --model resnet50 \
    --IPC 2 --loss PA --emb 128 \
    --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 5 --lr_decay_gamma 0.5 \
    --save_emb True

module purge
