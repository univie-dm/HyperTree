#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "hyperbolic_clustering"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=3-00:00:00
#SBATCH --error=logs/%j_%a.err
#SBATCH --output=logs/%j_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-96%16

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Note: WORLD_SIZE is set automatically by srun via SLURM_NTASKS;
# Lightning reads SLURM env vars for DDP coordination.

# ============================================================================
# Sweep grid: 96 configs — covers both pretraining AND clustering params
# for dendrogram purity optimization
# ============================================================================
# Axis 0: Backbone (2)                  - resnet50, resnet18
# Axis 1: Latent dim (3)                - 64, 128, 256
# Axis 2: Pretrain LR (2)               - 5e-4, 1e-3
# Axis 3: Contrastive weight (2)        - 0.0 (pure DINO), 0.5 (DINO + hyperbolic)
# Axis 4: Clustering temperature (2)    - (0.2→0.1), (0.3→0.05)  [sharper = more hierarchy]
# Axis 5: SNN k (2)                     - 10, 20                  [neighborhood for similarity]
# Total: 2 * 3 * 2 * 2 * 2 * 2 = 96

BACKBONES=(resnet50 resnet18)
LATENT_DIMS=(64 128 256)
PRETRAIN_LRS=(5e-4 1e-3)
CONTRASTIVE_WTS=(0.0 0.5)
START_TEMPS=(0.2 0.3)
END_TEMPS=(0.1 0.05)
SNN_KS=(10 20)

# Decode array index into grid coordinates
BB_IDX=$((SLURM_ARRAY_TASK_ID % 2))
LD_IDX=$(( (SLURM_ARRAY_TASK_ID / 2) % 3 ))
LR_IDX=$(( (SLURM_ARRAY_TASK_ID / 6) % 2 ))
CW_IDX=$(( (SLURM_ARRAY_TASK_ID / 12) % 2 ))
TEMP_IDX=$(( (SLURM_ARRAY_TASK_ID / 24) % 2 ))
SNN_IDX=$(( (SLURM_ARRAY_TASK_ID / 48) % 2 ))

BACKBONE=${BACKBONES[$BB_IDX]}
L_DIM=${LATENT_DIMS[$LD_IDX]}
P_LR=${PRETRAIN_LRS[$LR_IDX]}
C_WT=${CONTRASTIVE_WTS[$CW_IDX]}
S_TEMP=${START_TEMPS[$TEMP_IDX]}
E_TEMP=${END_TEMPS[$TEMP_IDX]}
SNN_K=${SNN_KS[$SNN_IDX]}

# Disable contrastive loss if weight is 0
if [ "$C_WT" = "0.0" ]; then
    C_ENABLED=False
else
    C_ENABLED=True
fi

echo "=== Array Task ${SLURM_ARRAY_TASK_ID} ==="
echo "Backbone=${BACKBONE} | LatentDim=${L_DIM} | PretrainLR=${P_LR} | ContrastiveWt=${C_WT}"
echo "ClusterTemp=${S_TEMP}→${E_TEMP} | SNN_k=${SNN_K}"

LOCAL_DATA="/local/imagenet"
if [ ! -d "$LOCAL_DATA" ]; then
    echo "Copying ImageNet to local SSD..."
    mkdir -p /local/
    cp -r $SCRATCH/imagenet /tmp/
fi

uv run train.py --config-name imagenet_dino \
    dataset.data_dir=/tmp/imagenet \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    experiment.seed=42 \
    architecture.model_type=${BACKBONE} \
    architecture.latent_dim=${L_DIM} \
    pretraining.epochs=60 \
    pretraining.learning_rate=${P_LR} \
    pretraining.accumulate_gradients=2 \
    pretraining.dino.warmup_teacher_temp_epochs=10 \
    pretraining.contrastive_loss.enabled=${C_ENABLED} \
    pretraining.contrastive_loss.weight=${C_WT} \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.learning_rate=1e-4 \
    clustering.accumulate_gradients=2 \
    clustering.hyperbolic_loss.start_temperature=${S_TEMP} \
    clustering.hyperbolic_loss.end_temperature=${E_TEMP} \
    clustering.hyperbolic_loss.weight_interval=20 \
    clustering.hyperbolic_loss.temperature_interval=20 \
    clustering.hyperbolic_loss.snn_k=${SNN_K} \
    clustering.lr_scheduler.type=warmup_cosine \
    clustering.lr_scheduler.interval=10 \
    experiment.log_interval=10 \
    evaluation_metrics.test_sample_size=10000

rm -rf /local/imagenet
