#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_h2h_rescue"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/inat18_h2h_rescue_%A_%a.err
#SBATCH --output=logs/inat18_h2h_rescue_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-8%8

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 rescue sweep — fix Poincaré collapse observed in the main sweep.
#
# Problem: in sbatch_vsc_inat18_h2h_dino_frozen.sh (main sweep, seeds 1-3 × 3
# conditions), ALL 6 Poincaré variants (warm+no_hyphc) collapsed to n@1≈0.001
# at epoch 29-50. The Euclidean baseline reached n@1=0.04 (gentle-star-1211).
# The 2 configs differ only by `architecture.manifold_name` — so the collapse
# is a Poincaré-specific failure, not a loss-schedule problem.
#
# Hypothesis: density-induced boundary crush. 8142 iNat classes packed into
# d=128 Poincaré ball at `curvature_magnitude=1.0` saturates the tanh in the
# HyperbolicLinearPP projection head, pushing all points to the boundary where
# they become indistinguishable. ImageNet has only 1000 classes in the same
# d=128 ball and Poincaré there does NOT collapse (young-night-1189 etc show
# progress at the same config).
#
# Fix: sweep `curvature_magnitude` — lower c means flatter geometry (closer to
# Euclidean at c→0), avoiding boundary saturation.  HIER uses c=0.1 for their
# Poincaré DML; we try three values.
#
# All conds use the WARM HypHC schedule (ew=5.0, st=1.0, et=0.5) and
# backbone_knn=True (what worked on ImageNet d=128). Only curvature varies.
#
# 3 curvatures × 3 seeds = 9 runs; same array size / throughput as main sweep.
# task_id = seed_idx*3 + cond_idx
#   seed_idx 0,1,2 -> seeds 1, 2, 3 (match main sweep for pairing)
#   cond 0: PoincareBall c=0.1  (HIER-style; very flat)
#   cond 1: PoincareBall c=0.3  (moderate)
#   cond 2: PoincareBall c=0.5  (halfway toward default)
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

CONDS=(
  "0.1:poincare_c01_warm"
  "0.3:poincare_c03_warm"
  "0.5:poincare_c05_warm"
)
SEEDS=(1 2 3)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 3))

IFS=':' read -r CURV CND_TAG <<< "${CONDS[$COND_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

echo "=== inat18_h2h_rescue task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} c=${CURV} ==="

# Stage data to local SSD if possible
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
    architecture.manifold_name=PoincareBall \
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
    wandb.tags='[inat18,h2h,dino_frozen,d128,paper_main,rescue,vsc,'${CND_TAG}',seed'${SEED}']'
