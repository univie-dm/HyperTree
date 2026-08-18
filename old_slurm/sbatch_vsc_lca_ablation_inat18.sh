#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "lca_abl_inat"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/lca_abl_inat_%A_%a.err
#SBATCH --output=logs/lca_abl_inat_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-5%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LCA-rectification ablation on iNat18. Uses the rescue-sweep winner config:
# PoincaréBall c=0.3 + HypHC warm, frozen FB DINO ResNet50, 50 ep clustering,
# bk=True. 2 LCA × 3 seeds = 6 runs.
#
# Reminder: 1-NN is at random (~0.0001) under this config — the iNat
# encoder is collapsed at the leaf level. The hDP_ward signal lives in
# the Ward dendrogram capturing high-level taxonomy. Frame as
# hierarchy-only in the paper.
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

LOCAL_DATA="/tmp/inat18"
if [ ! -d "$LOCAL_DATA" ] && [ -d "$DATA/inat18" ]; then
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/inat18/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="data/inat18"
fi

LCAS=("ours" "chami")
SEEDS=(1 2 3)

LCA_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))

LCA=${LCAS[$LCA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== lca_abl_inat task ${SLURM_ARRAY_TASK_ID}: lca=${LCA} seed=${SEED} c=0.3 ==="

uv run python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=inat18 \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=${SEED} \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.curvature_magnitude=0.3 \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.pretrained_weights_path=${DINO_CKPT} \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.lca_version=${LCA} \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.inat18_hierarchy_path='./data/inat18_taxonomy_hierarchy.txt' \
    wandb.tags='[inat18,lca_ablation,d128,c03_warm,bk1,lca_'${LCA}',seed'${SEED}',vsc]'
