#!/bin/bash
#SBATCH --job-name=imnet_d32
#SBATCH --time=2-00:00:00
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/imagenet_d32_%A_%a.log
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
# ImageNet-1k H2H at **latent_dim=32** — paper-framing sweep. The running
# d=128 sweep (sbatch_imagenet_h2h.sh) stays as the cross-dim ablation; this
# d=32 run is what the Section 5.4 main table should cite. CIFAR-100 showed
# +0.15 hDP_ward moving from d=128 → d=16; the paper's "exceptionally
# low-dimensional" headline requires low-d numbers on ImageNet too.
#
# Recipe choices (adapted from *-16-Ours CIFAR winner to frozen-backbone
# constraints of ImageNet):
#   - Frozen FB DINO ResNet50 backbone (unchanged — this is the SSL story)
#   - Contrastive enabled DURING clustering (weight=0.1, t=0.5) — only
#     trains the hyperbolic head since encoder is frozen
#   - HypHC ramp start_weight=1.0 → end_weight=5.0 (not 0→5) matches the
#     winning CIFAR recipe
#   - Cold temperature schedule (0.2 → 0.1)
#   - bk=True fixed (uniformly helped on STL-10 + CIFAR-equivalent runs)
#   - 3 seeds {1,2,3} to match *-16-Ours naming
#
# 3 conditions × 3 seeds × bk=True = 9 runs.
# task_idx = seed_idx*3 + cond_idx
#   cond 0: PoincareBall + HypHC(ew=5) cold -> poincare_hyphc_cold
#   cond 1: PoincareBall + HypHC(ew=0)      -> poincare_no_hyphc
#   cond 2: Euclidean    + HypHC(ew=0)      -> euclidean_baseline
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

CONDS=(
  "PoincareBall:5.0:poincare_hyphc_cold"
  "PoincareBall:0.0:poincare_no_hyphc"
  "Euclidean:0.0:euclidean_baseline"
)
SEEDS=(1 2 3)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 3))

IFS=':' read -r MANIFOLD HC_W CND_TAG <<< "${CONDS[$COND_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

# HypHC start_weight: 1.0 when HypHC is on (ramp 1→5), 0.0 when off
if [ "$HC_W" = "0.0" ]; then
    HC_START=0.0
else
    HC_START=1.0
fi

echo "=== imagenet_d32 task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} ==="

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=32 \
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
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=${HC_START} \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,h2h,dino_frozen,d32,paper_main,'${CND_TAG}',seed'${SEED}']'

module purge
