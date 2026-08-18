#!/bin/bash
#SBATCH --job-name=imnet_h2h_d16
#SBATCH --time=2-00:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/imagenet_h2h_d16_%A_%a.log
#SBATCH -A datamining
#SBATCH --exclude=vader,galadriel
#SBATCH --array=0-15%4

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# ImageNet-1k H2H sweep at latent_dim=16 — fills the gap between d=32
# (paper_main) and d=128 (H2H) so the latent-size axis matches CIFAR/TinyINet.
# Mirrors sbatch_imagenet_h2h.sh exactly except for latent_dim=16.
#
# 4 conditions x 2 backbone_knn x 2 seeds = 16 clustering-only runs.
#
# Array decoding:
#   task_id = seed_idx * 8  +  bk_idx * 4  +  cond_idx
#   cond_idx (0..3)  -> CONDS below
#   bk_idx   (0..1)  -> backbone_knn: False, True
#   seed_idx (0..1)  -> seeds 42, 43
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

CONDS=(
  "PoincareBall:5.0:1.0:0.5:poincare_hyphc_warm"
  "PoincareBall:5.0:0.2:0.1:poincare_hyphc_cold"
  "PoincareBall:0.0:0.2:0.1:poincare_no_hyphc"
  "Euclidean:0.0:0.2:0.1:euclidean_baseline"
)
SEEDS=(42 43)
BK_VALS=(False True)

COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))
BK_IDX=$(( (SLURM_ARRAY_TASK_ID / 4) % 2 ))
SEED_IDX=$(( (SLURM_ARRAY_TASK_ID / 8) % 2 ))

IFS=':' read -r MANIFOLD HC_W ST_T ET_T CND_TAG <<< "${CONDS[$COND_IDX]}"
BK_KNN=${BK_VALS[$BK_IDX]}
SEED=${SEEDS[$SEED_IDX]}
BK_TAG=$([ "$BK_KNN" = "True" ] && echo "bk1" || echo "bk0")

echo "=== imagenet_h2h_d16 task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} bk=${BK_KNN} d=16 ==="

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=16 \
    architecture.manifold_name=${MANIFOLD} \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=False \
    pretraining.pretrained_weights_path=${DINO_CKPT} \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.accumulate_gradients=2 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=${ST_T} \
    clustering.hyperbolic_loss.end_temperature=${ET_T} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=${BK_KNN} \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,h2h,dino_frozen,d16,'${CND_TAG}','${BK_TAG}',seed'${SEED}']'

module purge
