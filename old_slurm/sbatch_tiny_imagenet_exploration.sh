#!/bin/bash
#SBATCH --job-name=tinyimg_explore
#SBATCH --time=1-00:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/tinyimg_explore_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-11%6

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# TinyImageNet exploration sweep — 12 configs, 60 clustering epochs each
# Tests backbone fix (A), partial unfreeze (B), SimCLR vs DINO, crop sizes
# ============================================================================
# 0:  SimCLR r18, freeze
# 1:  SimCLR r50, freeze
# 2:  SimCLR r18, unfreeze layer4
# 3:  SimCLR r50, unfreeze layer4
# 4:  DINO r18, 100ep, 6crops 32px, freeze
# 5:  DINO r50, 100ep, 6crops 32px, freeze
# 6:  DINO r18, 200ep, 6crops 32px, freeze
# 7:  DINO r50, 200ep, 6crops 32px, freeze
# 8:  DINO r18, 100ep, 4crops 48px, freeze
# 9:  DINO r50, 100ep, 4crops 48px, freeze
# 10: DINO r18, 100ep, 6crops 32px, unfreeze layer4
# 11: DINO r50, 100ep, 6crops 32px, unfreeze layer4

IDX=$SLURM_ARRAY_TASK_ID

# Common clustering overrides (60 epoch quick test)
CLUSTER_ARGS="clustering.epochs=60 clustering.hyperbolic_loss.weight_interval=30 clustering.hyperbolic_loss.temperature_interval=30"

case $IDX in
  0)
    echo "=== SimCLR resnet18, freeze ==="
    python train.py --config-name tiny_imagenet_simclr \
      architecture.model_type=resnet18 \
      architecture.projection_layer_dims='[512,512]' \
      architecture.latent_dim=512 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,simclr,explore]'
    ;;
  1)
    echo "=== SimCLR resnet50, freeze ==="
    python train.py --config-name tiny_imagenet_simclr \
      architecture.model_type=resnet50 \
      architecture.projection_layer_dims='[1024,512]' \
      architecture.latent_dim=512 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,simclr,explore]'
    ;;
  2)
    echo "=== SimCLR resnet18, unfreeze layer4 ==="
    python train.py --config-name tiny_imagenet_simclr \
      architecture.model_type=resnet18 \
      architecture.projection_layer_dims='[512,512]' \
      architecture.latent_dim=512 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=layer4 \
      clustering.backbone_lr_factor=0.1 \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,simclr,explore]'
    ;;
  3)
    echo "=== SimCLR resnet50, unfreeze layer4 ==="
    python train.py --config-name tiny_imagenet_simclr \
      architecture.model_type=resnet50 \
      architecture.projection_layer_dims='[1024,512]' \
      architecture.latent_dim=512 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=layer4 \
      clustering.backbone_lr_factor=0.1 \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,simclr,explore]'
    ;;
  4)
    echo "=== DINO resnet18, 100ep, 6crops 32px, freeze ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet18 \
      architecture.projection_layer_dims='[512,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=100 \
      pretraining.n_local_crops=6 \
      pretraining.local_crop_size=32 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
  5)
    echo "=== DINO resnet50, 100ep, 6crops 32px, freeze ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet50 \
      architecture.projection_layer_dims='[1024,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=100 \
      pretraining.n_local_crops=6 \
      pretraining.local_crop_size=32 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
  6)
    echo "=== DINO resnet18, 200ep, 6crops 32px, freeze ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet18 \
      architecture.projection_layer_dims='[512,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=200 \
      pretraining.n_local_crops=6 \
      pretraining.local_crop_size=32 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
  7)
    echo "=== DINO resnet50, 200ep, 6crops 32px, freeze ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet50 \
      architecture.projection_layer_dims='[1024,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=200 \
      pretraining.n_local_crops=6 \
      pretraining.local_crop_size=32 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
  8)
    echo "=== DINO resnet18, 100ep, 4crops 48px, freeze ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet18 \
      architecture.projection_layer_dims='[512,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=100 \
      pretraining.n_local_crops=4 \
      pretraining.local_crop_size=48 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
  9)
    echo "=== DINO resnet50, 100ep, 4crops 48px, freeze ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet50 \
      architecture.projection_layer_dims='[1024,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=100 \
      pretraining.n_local_crops=4 \
      pretraining.local_crop_size=48 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=null \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
  10)
    echo "=== DINO resnet18, 100ep, 6crops 32px, unfreeze layer4 ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet18 \
      architecture.projection_layer_dims='[512,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=100 \
      pretraining.n_local_crops=6 \
      pretraining.local_crop_size=32 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=layer4 \
      clustering.backbone_lr_factor=0.1 \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
  11)
    echo "=== DINO resnet50, 100ep, 6crops 32px, unfreeze layer4 ==="
    python train.py --config-name tiny_imagenet_dino \
      architecture.model_type=resnet50 \
      architecture.projection_layer_dims='[1024,512]' \
      architecture.latent_dim=512 \
      pretraining.epochs=100 \
      pretraining.n_local_crops=6 \
      pretraining.local_crop_size=32 \
      clustering.freeze_encoder=True \
      clustering.unfreeze_backbone_from=layer4 \
      clustering.backbone_lr_factor=0.1 \
      $CLUSTER_ARGS \
      wandb.tags='[tiny_imagenet,dino,explore]'
    ;;
esac

module purge
