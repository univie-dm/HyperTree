#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "imnet_sched_fix"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/imnet_sched_fix_%A_%a.err
#SBATCH --output=logs/imnet_sched_fix_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --array=0-5%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# ImageNet schedule-overshoot fix verification.
#
# RESULTS.md flags that on ImageNet d=32, hyphc_warm peaks at hDP_ward=0.666
# at ce ≈ 18 then crashes to 0.444 by ce=50 — same boundary-saturation pattern
# as CIFAR-100 d=16. The CIFAR-100 fix sweep tested two knobs:
#   low_weight: end_weight=1.5 (was 5.0), weight_interval=20
#   flat_temp:  end_temperature=0.2 (was 0.1), no temperature ramp
#
# This sweep transfers both knobs to ImageNet d=128 hyphc_warm bk=0 (the H2H
# winner at 0.683 ± 0.012). If either knob holds the peak through ce=50, the
# paper's ImageNet headline lands at ~0.7 instead of needing a peak/final
# caveat.
#
# 2 schedule variants × 3 seeds = 6 runs.
#   variant 0: low_weight  (ew=1.5, weight_interval=20, st/et unchanged)
#   variant 1: flat_temp   (et=0.5 == st, no ramp; ew unchanged)
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

VARIANTS=(
  "1.5:1.0:0.5:20:low_weight"
  "5.0:1.0:0.5:250:flat_temp"
)
SEEDS=(42 43 44)

VAR_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))

IFS=':' read -r EW ST_T ET_T WINT VAR_TAG <<< "${VARIANTS[$VAR_IDX]}"

# flat_temp: hold temperature flat at start_temperature
if [ "$VAR_TAG" = "flat_temp" ]; then
    ET_T=${ST_T}
fi

SEED=${SEEDS[$SEED_IDX]}

echo "=== imnet_sched_fix task ${SLURM_ARRAY_TASK_ID}: ${VAR_TAG} seed=${SEED} ew=${EW} st=${ST_T} et=${ET_T} ==="

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
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${EW} \
    clustering.hyperbolic_loss.weight_interval=${WINT} \
    clustering.hyperbolic_loss.start_temperature=${ST_T} \
    clustering.hyperbolic_loss.end_temperature=${ET_T} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,schedule_fix,d128,h2h_warm,bk0,'${VAR_TAG}',seed'${SEED}',vsc]'
