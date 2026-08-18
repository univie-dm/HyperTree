#!/bin/bash
#SBATCH --job-name=imnet_dino
#SBATCH --time=2-00:00:00
#SBATCH -c 64
#SBATCH --mem=500G
#SBATCH --gres=gpu:4
#SBATCH --constraint='nvidia-h200'
#SBATCH --output=logs/imnet_dino_pipeline_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Two-stage fully self-supervised pipeline (DINO counterpart to the SimCLR
# pipeline in sbatch_imagenet_simclr_pipeline.sh):
#   Stage 1: DINO pretrain on full ImageNet, no labels (100 epochs)
#            4 H100/H200 GPUs, multi-crop (2 global + 6 local), batch 128/GPU
#            with grad-accum 2 -> effective batch ~1024
#            Checkpointing enabled — resumable across SLURM jobs
#   Stage 2: Transfer DINO features to CUB-200 hyperbolic clustering
#            Single-GPU clustering on the small CUB dataset (600 epochs)
# ============================================================================

echo "=== Stage 1: DINO pretrain on ImageNet (4 GPUs) ==="
uv run python train.py --config-name imagenet_dino_pretrain

# Resolve the saved pretrained model path (hash-based, includes method/aug now)
MODEL_PATH=$(uv run python -c "
from src.utils.utils import get_model_path
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
import os

with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
    cfg = compose(config_name='imagenet_dino_pretrain')
    print(f'data/models/{get_model_path(cfg)}.pt')
")

echo "=== Pretrained model expected at: ${MODEL_PATH} ==="

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Pretrained model not found at ${MODEL_PATH}"
    echo "Stage 1 likely did not complete. Re-submit to resume from checkpoint."
    exit 1
fi

echo "=== Stage 2: Transfer to CUB-200 hyperbolic clustering ==="
uv run python train.py --config-name cub200_dino_transfer \
    pretraining.pretrained_model_path="${MODEL_PATH}"

module purge
