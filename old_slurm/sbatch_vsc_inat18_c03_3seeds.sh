#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_c03"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/inat18_c03_%A_%a.err
#SBATCH --output=logs/inat18_c03_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-2%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 c=0.3 final headline rerun — only needed if you want a clean 3-seed
# row at the rescue winner with consistent tagging (`paper_main` instead of
# `rescue`). The 9 rescue runs on 2026-04-28 already give c=0.3 mean
# hDP_ward = 0.452 ± 0.106 across seeds {1,2,3} — those numbers are
# directly usable, so this rerun is OPTIONAL.
#
# Submit only if (a) you want consistent run tags for the paper headline,
# OR (b) you want to test more seeds (extend SEEDS array). Otherwise, use
# the existing rescue runs.
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

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

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
    clustering.hyperbolic_loss.lca_version=ours \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.inat18_hierarchy_path='./data/inat18_taxonomy_hierarchy.txt' \
    wandb.tags='[inat18,paper_main,d128,c03_warm,bk1,seed'${SEED}',vsc]'
