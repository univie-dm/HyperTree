#!/bin/bash
#SBATCH --job-name=tinyimg_sweep
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --output=logs/tinyimg_sweep_%A_%a.log
#SBATCH --array=0-47%16  # 48 jobs total, max 8 running concurrently

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# ============================================================================
# Sweep grid: 48 configs — SimCLR pretraining + hyperbolic clustering
# on Tiny ImageNet (200 classes, 64×64)
# ============================================================================
# Axis 0: Backbone (2)                  - resnet18, resnet50
# Axis 1: Latent dim + projection (3)   - 32/[256,128], 64/[512,256,128], 128/[512,256,128]
# Axis 2: Pretrain LR (2)              - 3e-3, 1e-3
# Axis 3: Clustering temperature (2)   - (0.2→0.1), (0.3→0.05)
# Axis 4: SNN k (2)                    - 10, 20
# Total: 2 * 3 * 2 * 2 * 2 = 48

BACKBONES=(resnet18 resnet50)
LATENT_DIMS=(32 64 128)
# Projection layer dims paired with latent dims (last proj dim >= latent dim)
PROJ_DIMS_0="[256,128]"        # for latent_dim=32
PROJ_DIMS_1="[512,256,128]"    # for latent_dim=64
PROJ_DIMS_2="[512,256,128]"    # for latent_dim=128
PRETRAIN_LRS=(3e-3 1e-3)
START_TEMPS=(0.2 0.3)
END_TEMPS=(0.1 0.05)
SNN_KS=(10 20)

# Decode array index into grid coordinates
BB_IDX=$((SLURM_ARRAY_TASK_ID % 2))
LD_IDX=$(( (SLURM_ARRAY_TASK_ID / 2) % 3 ))
LR_IDX=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))
TEMP_IDX=$(( (SLURM_ARRAY_TASK_ID / 12) % 2 ))
SNN_IDX=$(( (SLURM_ARRAY_TASK_ID / 24) % 2 ))

BACKBONE=${BACKBONES[$BB_IDX]}
L_DIM=${LATENT_DIMS[$LD_IDX]}
P_LR=${PRETRAIN_LRS[$LR_IDX]}
S_TEMP=${START_TEMPS[$TEMP_IDX]}
E_TEMP=${END_TEMPS[$TEMP_IDX]}
SNN_K=${SNN_KS[$SNN_IDX]}

# Select projection dims based on latent dim index
case $LD_IDX in
    0) PROJ_DIMS=$PROJ_DIMS_0 ;;
    1) PROJ_DIMS=$PROJ_DIMS_1 ;;
    2) PROJ_DIMS=$PROJ_DIMS_2 ;;
esac

cp -r $SCRATCH/tiny-imagenet-200 /tmp/


echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "Backbone=${BACKBONE} | LatentDim=${L_DIM} | ProjDims=${PROJ_DIMS} | PretrainLR=${P_LR}"
echo "ClusterTemp=${S_TEMP}→${E_TEMP} | SNN_k=${SNN_K}"

uv run python train.py --config-name tiny_imagenet_simclr \
    dataset.data_dir=/tmp/tiny-imagenet-200 \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    experiment.seed=42 \
    architecture.model_type=${BACKBONE} \
    architecture.latent_dim=${L_DIM} \
    "architecture.projection_layer_dims=${PROJ_DIMS}" \
    pretraining.epochs=400 \
    pretraining.learning_rate=${P_LR} \
    clustering.epochs=300 \
    clustering.freeze_encoder=True \
    clustering.learning_rate=1e-4 \
    clustering.hyperbolic_loss.start_temperature=${S_TEMP} \
    clustering.hyperbolic_loss.end_temperature=${E_TEMP} \
    clustering.hyperbolic_loss.weight_interval=150 \
    clustering.hyperbolic_loss.temperature_interval=150 \
    clustering.hyperbolic_loss.snn_k=${SNN_K} \
    clustering.lr_scheduler.type=warmup_cosine \
    clustering.lr_scheduler.interval=20 \
    experiment.log_interval=10
