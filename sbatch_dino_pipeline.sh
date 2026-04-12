#!/bin/bash
#SBATCH --job-name=dino_pipeline
#SBATCH --time=1-00:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/dino_pipeline_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Two-stage fully self-supervised pipeline:
#   Stage 1: DINO pretrain on TinyImageNet (100 epochs, ~3-4h on H100)
#            Also clusters on TinyImageNet for TinyImageNet results
#   Stage 2: Transfer DINO features to CUB-200 clustering (600 epochs, ~2-3h)
# ============================================================================

echo "=== Stage 1: DINO pretrain + cluster on TinyImageNet ==="
python train.py --config-name tiny_imagenet_dino pretraining.accumulate_gradients=2

# Find the pretrained model path (hash-based)
MODEL_PATH=$(python -c "
from src.utils.utils import get_model_path
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize_config_dir
import os

with initialize_config_dir(config_dir=os.path.abspath('configs'), version_base=None):
    cfg = compose(config_name='tiny_imagenet_dino')
    print(f'data/models/{get_model_path(cfg)}.pt')
")

echo "=== Pretrained model at: ${MODEL_PATH} ==="

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Pretrained model not found at ${MODEL_PATH}"
    exit 1
fi

echo "=== Stage 2: Transfer to CUB-200 clustering ==="
python train.py --config-name cub200_dino_transfer \
    pretraining.pretrained_model_path="${MODEL_PATH}"

module purge
