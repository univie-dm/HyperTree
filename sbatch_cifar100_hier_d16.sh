#!/bin/bash
#SBATCH --job-name=hier_c100_d16
#SBATCH --time=16:00:00
#SBATCH -c 16
#SBATCH --mem=164G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/hier_cifar100_d16_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-2%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# ============================================================================
# HIER CIFAR-100 at emb=16, 3 seeds for error bars on the matched-dim
# supervised baseline. Identical recipe to sbatch_cifar100_hier.sh except
# --emb 16 and --run_name. Eval via sbatch_eval_hier_cifar100_d16.sh after.
# ============================================================================

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== hier_c100_d16 task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

cd /srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/HIER-CVPR23

mkdir -p logs/CIFAR100

uv run torchrun --standalone --nproc_per_node=1 hier/train.py \
    --data_path /srv/home/users/sidakk95cs/hyperbolic-clustering/data \
    --output_dir ./logs \
    --run_name seed${SEED}_emb16 \
    --seed ${SEED} \
    --lr 1e-4 --epochs 100 --batch_size 90 \
    --num_hproxies 512 --clip_r 2.3 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 5 \
    --lambda1 1 --lambda2 0.5 --topk 20 --mrg 0.1 \
    --dataset CIFAR100 --model resnet50 \
    --IPC 2 --loss PA --emb 16 \
    --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 5 --lr_decay_gamma 0.5 \
    --save_emb True

module purge
