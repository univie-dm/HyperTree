#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_sched_fix"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/inat18_sched_fix_%A_%a.err
#SBATCH --output=logs/inat18_sched_fix_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-5

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 schedule-overshoot-fix transfer.
#
# Problem: at iNat18 c=0.3 d=128 paper_main (warm HypHC st=1.0/et=0.5,
# ew=5.0), final hDP_ward = 0.872 ± 0.021 but Δ from ce=0 is only ~+0.04 —
# clustering training contributes little above the random-init head.
# Same warm schedule on ImageNet d=128 crashed peak→final by 0.24
# hDP_ward; the schedule_fix sweep there (low_weight ew=1.5/wint=20 and
# flat_temp et=st) recovered +0.35 over the crashed baseline (0.444 → 0.789).
# This sweep transfers both knobs to iNat at c=0.3.
#
# 2 schedule variants × 3 seeds = 6 runs.
#   variant 0: low_weight  (ew=1.5, weight_interval=20, st/et unchanged)
#   variant 1: flat_temp   (et=1.0 == st, no temp ramp; ew unchanged at 5.0)
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

VARIANTS=(
  "1.5:1.0:0.5:20:low_weight"
  "5.0:1.0:0.5:250:flat_temp"
)
SEEDS=(1 2 3)

VAR_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))

IFS=':' read -r EW ST_T ET_T WINT VAR_TAG <<< "${VARIANTS[$VAR_IDX]}"

# flat_temp: hold temperature flat at start_temperature (1.0)
if [ "$VAR_TAG" = "flat_temp" ]; then
    ET_T=${ST_T}
fi

SEED=${SEEDS[$SEED_IDX]}

echo "=== inat18_sched_fix task ${SLURM_ARRAY_TASK_ID}: ${VAR_TAG} seed=${SEED} ew=${EW} st=${ST_T} et=${ET_T} wint=${WINT} ==="

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
    clustering.hyperbolic_loss.end_weight=${EW} \
    clustering.hyperbolic_loss.weight_interval=${WINT} \
    clustering.hyperbolic_loss.start_temperature=${ST_T} \
    clustering.hyperbolic_loss.end_temperature=${ET_T} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.inat18_hierarchy_path='./data/inat18_taxonomy_hierarchy.txt' \
    wandb.tags='[inat18,schedule_fix,d128,c03_warm,bk1,'${VAR_TAG}',seed'${SEED}',vsc]'
