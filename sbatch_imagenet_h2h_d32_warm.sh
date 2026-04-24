#!/bin/bash
#SBATCH --job-name=imnet_d32_w
#SBATCH --time=2-00:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/imagenet_d32_warm_%A_%a.log
#SBATCH -A datamining
#SBATCH --exclude=vader,galadriel
#SBATCH --array=0-2%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# ImageNet-1k d=32 **warm HypHC** follow-up to sbatch_imagenet_h2h_d32.sh.
#
# Why this exists: the d=32 sweep only ran cold HypHC (st=0.2, et=0.1) but
# seed=1 results show cold underperforming no_hyphc at d=32 (hDP_w 0.50 vs
# 0.56), mirroring the d=128 pattern where warm (eager-deluge-1114, hDP_w
# 0.675) clearly beat cold (0.44). TI also prefers warm (hDP_w 0.715 at
# st=1.0/et=0.5). So d=32 very likely wants warm too and the cold variant
# won't give us a representative paper number.
#
# This sbatch only runs the missing condition (Poincaré + HypHC warm) across
# 3 seeds; no_hyphc and euclidean_baseline are already covered by the cold
# sweep (same manifold, HypHC is off, schedule is irrelevant for them).
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== imagenet_d32_warm task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=32 \
    architecture.manifold_name=PoincareBall \
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
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,h2h,dino_frozen,d32,paper_main,poincare_hyphc_warm,seed'${SEED}']'

module purge
