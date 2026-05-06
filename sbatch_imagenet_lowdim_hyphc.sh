#!/bin/bash
#SBATCH --job-name=imnet_lowdim_hyphc
#SBATCH --time=2-00:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/imnet_lowdim_hyphc_%A_%a.log
#SBATCH -A datamining
#SBATCH --exclude=vader,galadriel
#SBATCH --array=0-8%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# ImageNet-1k low-dim HypHC dim ablation: d ∈ {16, 32, 64} × seed ∈ {1, 2, 3}
# Single condition: PoincareBall + HypHC cold (ramp 1→5, t 0.2→0.1, bk=True).
# Recipe matches the d=32 paper_main hyphc_cold winner (sbatch_imagenet_h2h_d32.sh)
# so the new low-dim numbers are directly comparable to existing d=32/d=128 rows.
#
# task_idx = seed_idx * 3 + dim_idx
#   dim 0: latent_dim=16
#   dim 1: latent_dim=32
#   dim 2: latent_dim=64
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

DIMS=(16 32 64)
SEEDS=(1 2 3)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
DIM_IDX=$((SLURM_ARRAY_TASK_ID % 3))
LATENT_DIM=${DIMS[$DIM_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CND_TAG="poincare_hyphc_cold"

echo "=== imnet_lowdim_hyphc task ${SLURM_ARRAY_TASK_ID}: d=${LATENT_DIM} seed=${SEED} ==="

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=${LATENT_DIM} \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=False \
    pretraining.pretrained_weights_path=${DINO_CKPT} \
    clustering.epochs=35 \
    clustering.freeze_encoder=True \
    clustering.accumulate_gradients=2 \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,h2h,dino_frozen,d'${LATENT_DIM}',lowdim_hyphc,paper_main,'${CND_TAG}',seed'${SEED}']'

module purge
