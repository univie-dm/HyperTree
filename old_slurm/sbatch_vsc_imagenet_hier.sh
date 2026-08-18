#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "hier_imagenet"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=2-00:00:00
#SBATCH --error=logs/hier_imagenet_%A_%a.err
#SBATCH --output=logs/hier_imagenet_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-2%2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# HIER on ImageNet-1k as a supervised DML baseline for the paper.
#
# Protocol: full-class split (all 1000 classes in train and eval) — matches
# our unsupervised clustering pipeline (1.28M train + 50k val, eval on the
# same 10k stratified subsample we use for "Ours").
#
# Compute is heavy — ResNet50 on 1.28M images at 224×224. We cap at 30
# epochs (HIER converges fast on CIFAR-100 and TI; 30ep is enough to peak
# given LR decay every 5 ep) and use 4 GPUs with effective batch = 4×90.
#
# 3-seed array (seeds 1, 2, 3) at d=128 (matches our ImageNet d=128 main row).
# Eval embeddings saved to baselines/HIER-CVPR23/logs/ImageNet/ for the
# scripts/eval_baseline_embeddings.py pipeline.
# ============================================================================

cd /srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/HIER-CVPR23

mkdir -p logs/ImageNet

# Stage data to local SSD if possible (HIER needs the imagenet/ subtree
# including ILSVRC2012_devkit_t12.tar.gz at the imagenet/ root)
LOCAL_DATA="/tmp/imagenet_hier_root"
if [ ! -d "$LOCAL_DATA/imagenet/val" ] && [ -d "$DATA/imagenet" ]; then
    echo "Copying ImageNet (incl. devkit) to local SSD..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/imagenet" "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA/imagenet/val" ]; then
    DATA_PATH=${LOCAL_DATA}
else
    DATA_PATH="$DATA"
fi

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

uv run torchrun --standalone --nproc_per_node=4 hier/train.py \
    --data_path ${DATA_PATH} \
    --output_dir ./logs \
    --run_name seed${SEED}_emb128 \
    --seed ${SEED} \
    --lr 1e-4 --epochs 30 --batch_size 90 \
    --num_hproxies 1024 --clip_r 2.3 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 3 \
    --lambda1 1 --lambda2 0.5 --topk 20 --mrg 0.1 \
    --dataset ImageNet --model resnet50 \
    --IPC 2 --loss PA --emb 128 \
    --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 3 --lr_decay_gamma 0.5 \
    --save_emb True
