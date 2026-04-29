#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "lca_abl_imnet"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/lca_abl_imnet_%A_%a.err
#SBATCH --output=logs/lca_abl_imnet_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --array=0-5%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LCA-rectification ablation on ImageNet d=128. Uses the d=128 H2H winner:
# PoincaréBall + HypHC warm, bk=0, frozen FB DINO ResNet50, 50 ep clustering.
# 2 LCA × 3 seeds = 6 runs.
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

LOCAL_DATA="/tmp/imagenet"
if [ ! -d "$LOCAL_DATA/train" ] && [ -d "$DATA/imagenet" ]; then
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/imagenet/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA/train" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="$DATA/imagenet"
fi

LCAS=("ours" "chami")
SEEDS=(42 43 44)

LCA_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))

LCA=${LCAS[$LCA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== lca_abl_imnet task ${SLURM_ARRAY_TASK_ID}: lca=${LCA} seed=${SEED} ==="

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.pretrained_weights_path=${DINO_CKPT} \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.accumulate_gradients=2 \
    clustering.hyperbolic_loss.lca_version=${LCA} \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,lca_ablation,d128,h2h_warm,bk0,lca_'${LCA}',seed'${SEED}',vsc]'
