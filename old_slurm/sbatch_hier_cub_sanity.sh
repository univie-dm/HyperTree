#!/bin/bash
#SBATCH --job-name=hier_cub
#SBATCH --time=08:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/hier_cub_sanity_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# HIER sanity run — reproduce their CUB R@1 ≈ 0.70 as a pre-flight check before
# adapting HIER to CIFAR-100 as a paper baseline. Matches their published
# scripts/Resnet50/hier_CUB.sh recipe but with:
#   - 1 GPU instead of 2 (our cluster scheduler is happier with 1 GPU H100)
#   - torchrun instead of deprecated torch.distributed.launch
# Expected: R@1 ≈ 0.70 at epoch 100. If we land at 0.02 or 0.95, something
# is broken in our env (newer timm/torch vs HIER's pins) and we need to debug
# BEFORE trusting any CIFAR-100 number from this pipeline.
# ============================================================================

cd /srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/HIER-CVPR23

mkdir -p logs/CUB

uv run torchrun --standalone --nproc_per_node=1 hier/train.py \
    --data_path /srv/home/users/sidakk95cs/hyperbolic-clustering/data \
    --output_dir ./logs \
    --run_name sanity_seed0 \
    --seed 0 \
    --lr 1e-4 --epochs 100 --batch_size 90 \
    --num_hproxies 512 --clip_r 2.3 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --lambda1 1 --lambda2 0.5 --topk 20 --mrg 0.1 \
    --dataset CUB --model resnet50 \
    --IPC 2 --loss PA --emb 512 \
    --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 5 --lr_decay_gamma 0.5

module purge
