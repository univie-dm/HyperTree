#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_h2h_dino"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/inat18_h2h_dino_%A_%a.err
#SBATCH --output=logs/inat18_h2h_dino_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-8%4

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 H2H sweep on VSC (4xH100) — FROZEN FB DINO ResNet50 backbone.
#
# Why this path (vs the existing sbatch_vsc_inat18_h2h.sh that does from-scratch
# DINO pretraining):
#   - From-scratch DINO on iNat18 (437k images) is ~weeks on VSC; the running
#     VSC SimCLR/DINO iNat pretrains sit at epoch 60-120 of 800/100 and won't
#     land inside the 10-day deadline.
#   - FB's DINO ResNet50 checkpoint is publicly-released SSL weights trained on
#     ImageNet. iNat18 ⊂ natural-image domain → reasonable transfer.
#   - This is the same frozen-backbone recipe that is producing ImageNet
#     hDP_ward=0.6755 at epoch 48/50 on the local cluster (job 533641_0,
#     eager-deluge-1114, poincare_hyphc_WARM).
#
# 3 conditions × 3 seeds = 9 runs; bk=True fixed (helped on TI/CIFAR memory).
# task_id = seed_idx*3 + cond_idx
#   seed_idx 0,1,2 -> seeds 1, 2, 3 (match *-*-Ours naming)
#   cond 0: PoincareBall + HypHC(ew=5) WARM  -> poincare_hyphc_warm
#   cond 1: PoincareBall + HypHC(ew=0)       -> poincare_no_hyphc
#   cond 2: Euclidean    + HypHC(ew=0)       -> euclidean_baseline
#
# Warm schedule (st=1.0, et=0.5) chosen over cold because the ImageNet d=128
# run shows WARM clearly ahead at epoch 48 (hDP 0.68 vs cold/no-hyphc still
# pre-divergence). iNat18's deeper taxonomy (7 levels) should respond similarly.
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

CONDS=(
  "PoincareBall:5.0:1.0:0.5:poincare_hyphc_warm"
  "PoincareBall:0.0:0.2:0.1:poincare_no_hyphc"
  "Euclidean:0.0:0.2:0.1:euclidean_baseline"
)
SEEDS=(1 2 3)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 3))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 3))

IFS=':' read -r MANIFOLD HC_W ST_T ET_T CND_TAG <<< "${CONDS[$COND_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

echo "=== inat18_h2h_dino task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} ==="

# Stage data to local SSD if possible (same pattern as sbatch_vsc_inat18_h2h.sh)
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
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.spectral_clustering=False \
    evaluation_metrics.test_sample_size=0 \
    wandb.tags='[inat18,h2h,dino_frozen,d128,paper_main,vsc,'${CND_TAG}',seed'${SEED}']'
