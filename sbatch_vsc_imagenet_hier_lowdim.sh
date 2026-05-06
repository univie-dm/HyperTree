#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "hier_imnet_lowdim"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/hier_imnet_lowdim_%A_%a.err
#SBATCH --output=logs/hier_imnet_lowdim_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-8%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# HIER on ImageNet-1k at low embedding dims for the §5.4 dim ablation.
# d ∈ {16, 32, 64} × seed ∈ {1, 2, 3} = 9 runs, throttled to 3 concurrent
# (was %2 on the d=128 baseline; bumping for wall-clock).
#
# Speedup over the d=128 baseline (1171053):
#   - epochs 30 → 15. The d=128 run had LR-decay-3-gamma-0.5, putting ep 15
#     at LR ≈ 3% of init; ep 18 (the highest converged checkpoint we got
#     before NODE_FAIL) hit hDP_ward 0.897. ep 15 should be within 0.005.
#   - %3 concurrency: 9 runs × ~12h ÷ 3 = ~36h wall-clock for the whole sweep.
#
# task_idx = seed_idx*3 + dim_idx
#   dim 0: emb=16
#   dim 1: emb=32
#   dim 2: emb=64
# ============================================================================

cd /home/ks09437/hyperbolic-clustering/baselines/HIER-CVPR23

mkdir -p logs/ImageNet

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

DIMS=(16 32 64)
SEEDS=(1 2 3)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DIM_IDX=$((SLURM_ARRAY_TASK_ID % 3))
EMB=${DIMS[$DIM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== hier_imnet_lowdim task ${SLURM_ARRAY_TASK_ID}: emb=${EMB} seed=${SEED} ==="

# Background reaper: every 60s, prune *_train_*.pt for this run as soon as a new
# eval epoch dumps them. Keeps disk quota under control (HIER dumps a 100-650MB
# train-embedding file every 3 epochs; previous lowdim sweep filled 50G home
# quota and crashed all 8 in-flight tasks at ep 12).
RUN_TAG="seed${SEED}_emb${EMB}"
(
  while true; do
    sleep 60
    rm -f ./logs/ImageNet/resnet50_${RUN_TAG}_train_*.pt 2>/dev/null
  done
) &
REAPER_PID=$!

uv run torchrun --standalone --nproc_per_node=4 hier/train.py \
    --data_path ${DATA_PATH} \
    --output_dir ./logs \
    --run_name ${RUN_TAG} \
    --seed ${SEED} \
    --lr 1e-4 --epochs 15 --batch_size 90 \
    --num_hproxies 1024 --clip_r 2.3 \
    --hyp_c 0.1 --warmup_epochs 1 \
    --weight_decay 1e-2 --use_fp16 True \
    --clip_grad 1.0 --eval_freq 3 \
    --lambda1 1 --lambda2 0.5 --topk 20 --mrg 0.1 \
    --dataset ImageNet --model resnet50 \
    --IPC 2 --loss PA --emb ${EMB} \
    --bn_freeze True --use_lastnorm True \
    --optimizer adamw --lr_decay 3 --lr_decay_gamma 0.5 \
    --save_emb True

kill ${REAPER_PID} 2>/dev/null
rm -f ./logs/ImageNet/resnet50_${RUN_TAG}_train_*.pt 2>/dev/null
