#!/bin/bash
#SBATCH --job-name "hc_inat18"
#SBATCH --gres=gpu:2
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=0-01:00:00
#SBATCH --error=logs/inat18_dino_%j.err
#SBATCH --output=logs/inat18_dino_%j.out
#SBATCH --array=0-95  # 96 jobs total

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# ============================================================================
# Sweep grid: 96 configs — iNat18-specific values
# ============================================================================
# iNat18 has 8,142 species (vs ImageNet 1K) with a 7-level deep taxonomy
# and extreme long-tail class imbalance. This motivates:
#   - Higher latent dims (128-512) for the 8x more classes
#   - Sharper temperature schedules to resolve the deeper hierarchy
#   - Smaller SNN k to protect rare species from being swamped by common ones
#
# Axis 0: Backbone (2)                  - resnet50, resnet18
# Axis 1: Latent dim (3)               - 128, 256, 512     [higher: 8K classes need capacity]
# Axis 2: Pretrain LR (2)              - 5e-4, 1e-3
# Axis 3: Contrastive weight (2)       - 0.0 (pure DINO), 0.5 (DINO + hyperbolic)
# Axis 4: Clustering temperature (2)   - (0.15→0.05), (0.3→0.05)  [sharper for deep taxonomy]
# Axis 5: SNN k (2)                    - 5, 15              [smaller for long-tail]
# Total: 2 * 3 * 2 * 2 * 2 * 2 = 96

BACKBONES=(resnet50 resnet18)
LATENT_DIMS=(128 256 512)
PRETRAIN_LRS=(5e-4 1e-3)
CONTRASTIVE_WTS=(0.0 0.5)
START_TEMPS=(0.15 0.3)
END_TEMPS=(0.05 0.05)
SNN_KS=(5 15)

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

uv run train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=inat18 \
    experiment.seed=42 \
    architecture.model_type=${BACKBONE} \
    architecture.latent_dim=${L_DIM} \
    pretraining.epochs=2 \
    pretraining.learning_rate=${P_LR} \
    pretraining.accumulate_gradients=4 \
    pretraining.dino.warmup_teacher_temp_epochs=10 \
    pretraining.contrastive_loss.enabled=${C_ENABLED} \
    pretraining.contrastive_loss.weight=${C_WT} \
    clustering.epochs=2 \
    clustering.freeze_encoder=True \
    clustering.learning_rate=1e-4 \
    clustering.accumulate_gradients=4 \
    clustering.hyperbolic_loss.start_temperature=${S_TEMP} \
    clustering.hyperbolic_loss.end_temperature=${E_TEMP} \
    clustering.hyperbolic_loss.weight_interval=20 \
    clustering.hyperbolic_loss.temperature_interval=20 \
    clustering.hyperbolic_loss.snn_k=${SNN_K} \
    clustering.lr_scheduler.type=warmup_cosine \
    clustering.lr_scheduler.interval=10 \
    experiment.log_interval=10 \
    evaluation_metrics.test_sample_size=10000
