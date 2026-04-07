#!/bin/bash
#SBATCH --job-name=hyper_sweep
#SBATCH --time=2-00:00:00
#SBATCH -c 32
#SBATCH --mem=350G
#SBATCH --gres=gpu:2
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/sweep_%A_%a.log
#SBATCH -x galadriel
#SBATCH -A datamining
#SBATCH --array=0-53%4  # 54 jobs total, max 4 running concurrently
    
export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

# Define hyperparameter possibilities
C_LRS=(1e-4 5e-5 1e-5)                          # 3 Clustering learning rates
CONTRASTIVE_WEIGHTS=(0.0 0.1 0.5)               # 3 Anchor weights
START_TEMPS=(0.1 0.3)                           # 2 Hyperbolic temperatures
LATENT_DIMS=(64 128 256)                        # 3 Latent dimensions

# Calculate indices for the 4D grid
LR_IDX=$((SLURM_ARRAY_TASK_ID % 3))
CW_IDX=$(( (SLURM_ARRAY_TASK_ID / 3) % 3 ))
TEMP_IDX=$(( (SLURM_ARRAY_TASK_ID / 9) % 2 ))
LD_IDX=$(( (SLURM_ARRAY_TASK_ID / 18) % 3 ))

# Retrieve the actual parameters
C_LR=${C_LRS[$LR_IDX]}
C_WEIGHT=${CONTRASTIVE_WEIGHTS[$CW_IDX]}
S_TEMP=${START_TEMPS[$TEMP_IDX]}
L_DIM=${LATENT_DIMS[$LD_IDX]}

echo "Starting Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Parameters: LR=${C_LR} | ContrastiveWeight=${C_WEIGHT} | StartTemp=${S_TEMP} | LatentDim=${L_DIM}"

uv run python train.py --config-name imagenet_poincare +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} dataset=imagenet \
    experiment.seed=69 \
    pretraining.load_supervised_weights=True \
    architecture.latent_dim=${L_DIM} \
    architecture.cluster_dim=${L_DIM} \
    pretraining.epochs=100 \
    pretraining.learning_rate=1e-3 \
    pretraining.accumulate_gradients=4 \
    pretraining.contrastive_loss.temperature=0.2 \
    pretraining.lr_scheduler.type=warmup_cosine \
    pretraining.lr_scheduler.interval=10 \
    clustering.epochs=40 \
    clustering.freeze_encoder=True \
    clustering.learning_rate=${C_LR} \
    clustering.contrastive_loss.weight=${C_WEIGHT} \
    clustering.hyperbolic_loss.start_temperature=${S_TEMP} \
    clustering.accumulate_gradients=4 \
    clustering.lr_scheduler.type=warmup_cosine \
    clustering.lr_scheduler.interval=30 \
    experiment.log_interval=20 \
    evaluation_metrics.test_sample_size=10000

module purge
