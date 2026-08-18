#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_finetune"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/inat18_finetune_%A_%a.err
#SBATCH --output=logs/inat18_finetune_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-5%6

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 backbone-finetune sweep — direct test of "is the bottleneck the
# backbone, the head, or both?"
#
# All prior iNat sweeps (h2h, h2h_rescue) freeze the encoder. Best result is
# Euclidean frozen at n@1=0.0103 (confused-gorge-1164); every Poincaré
# variant collapses (n@1 < 0.01) regardless of curvature in {0.1, 0.3, 0.5}.
# The frozen-DINOv2-ViT-L sbatch tests the backbone-quality side; this one
# tests whether adapting our existing FB DINO ResNet50 to iNat fixes it.
#
# Setup mirrors h2h_rescue.sh (FB DINO weights, 50 clustering epochs, warm
# HypHC schedule, backbone_knn=True), but with clustering.freeze_encoder=True
# AND clustering.unfreeze_backbone_from=layer4 — keeps layer1-3 frozen and
# trains layer4 + head + projection. backbone_lr_factor=0.1 (default) gives
# layer4 a 10× smaller LR than the head.
#
# Conds × seeds = 2 × 3 = 6 runs:
#   cond 0: Euclidean, unfreeze layer4
#   cond 1: PoincareBall c=0.1, unfreeze layer4 (HIER-style flat geometry)
# Pair with rescue results (frozen, c=0.1) and main h2h (frozen Euclidean,
# confused-gorge-1164) to isolate the unfreezing effect at matched seeds.
#
# task_id = seed_idx*2 + cond_idx
#   seed_idx 0,1,2 -> seeds 1, 2, 3 (matches rescue seed pairing)
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

CONDS=(
  "Euclidean:0.0:euclidean_layer4"
  "PoincareBall:0.1:poincare_c01_layer4"
)
SEEDS=(1 2 3)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 2))

IFS=':' read -r MANIFOLD CURV CND_TAG <<< "${CONDS[$COND_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

echo "=== inat18_finetune task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} c=${CURV} ==="

LOCAL_DATA="/tmp/inat18"
if [ ! -d "$LOCAL_DATA" ] && [ -d "$DATA/inat18" ]; then
    echo "Copying iNat18 to local SSD..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/inat18/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="data/inat18"
fi

uv run python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    experiment.num_workers=6 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=${MANIFOLD} \
    architecture.curvature_magnitude=${CURV} \
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
    clustering.unfreeze_backbone_from=layer4 \
    clustering.backbone_lr_factor=0.1 \
    clustering.accumulate_gradients=2 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.spectral_clustering=False \
    evaluation_metrics.test_sample_size=0 \
    wandb.tags='[inat18,h2h,dino_frozen,d128,paper_main,finetune,vsc,'${CND_TAG}',seed'${SEED}']'
