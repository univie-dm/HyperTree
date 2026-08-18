#!/bin/bash
#SBATCH --job-name=hier_c100
#SBATCH --time=16:00:00
#SBATCH -c 16
#SBATCH --mem=164G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/hier_cifar100_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# HIER on CIFAR-100 as a supervised DML baseline for the paper.
#
# Protocol: full-class split (all 100 classes in train and eval) — NOT HIER's
# native disjoint-class DML protocol. This is a head-to-head comparison with
# our unsupervised clustering method on the standard CIFAR-100 benchmark
# (50k train + 10k test).
#
# Framing: HIER uses class labels; our method does not. So R@1 is a
# supervised upper-bound reference. The interesting comparison is hDP_ward
# and dendrogram purity on HIER's embeddings (pipe through
# scripts/eval_baseline_embeddings.py after training) — HIER does not
# optimize hierarchical structure, so this is where our method can win.
#
# Hyperparams match HIER's published CUB recipe (scripts/Resnet50/hier_CUB.sh)
# with emb=128 instead of 512 to match our main-table CIFAR-100 dimension
# for the clustering eval.
#
# CUB sanity (job 534388) hit R@1 ≈ 0.68-0.69 (paper: 0.70) so env is OK.
# ============================================================================

cd /srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/HIER-CVPR23

mkdir -p logs/CIFAR100

uv run torchrun --standalone --nproc_per_node=1 hier/train.py \
    --data_path /srv/home/users/sidakk95cs/hyperbolic-clustering/data \
    --output_dir ./logs \
    --run_name seed1_emb128 \
    --seed 1 \
    --lr 1e-4 --epochs 100 --batch_size 90 \
    --num_hproxies 512 --clip_r 2.3 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --lambda1 1 --lambda2 0.5 --topk 20 --mrg 0.1 \
    --dataset CIFAR100 --model resnet50 \
    --IPC 2 --loss PA --emb 128 \
    --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 5 --lr_decay_gamma 0.5 \
    --save_emb True

module purge
