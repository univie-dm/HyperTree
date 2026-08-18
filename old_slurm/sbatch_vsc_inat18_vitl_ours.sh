#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_vitl_ours"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-12:00:00
#SBATCH --error=logs/inat18_vitl_ours_%A_%a.err
#SBATCH --output=logs/inat18_vitl_ours_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-1%2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 "Ours" with frozen DINOv2 ViT-L/14 backbone @ res 336.
#
# Goal: beat the current §5.3 number by upgrading the backbone from RN50-DINO
# (2021, ImageNet-1k SSL) to DINOv2-ViT-L/14 (2024, LVD-142M). Frozen-only
# baseline at res=336 is hDP_ward=0.851 (sweep job 1170346 task 0).
# Current ours-RN50 hits ~0.89 — if our pipeline lifts ViT-L features the
# same way it lifts RN50, we should land ≥0.91.
#
# CAVEAT (project_inat_poincare_density_collapse + ViT-B viability runs):
#   - ViT-B + clustering pipeline collapsed to hDP=0.30-0.45 (job 1170370/1170393)
#   - Likely cause: random-init projection head can't recover ViT features in
#     50 ep of clustering. Mitigation: 30 ep of contrastive pretraining of the
#     head BEFORE clustering, on the frozen backbone.
#
# Configs:
#   0: dinov2_vitl14, Euclidean, c=0.0  (HC disabled)
#   1: dinov2_vitl14, PoincareBall, c=0.1  (matches working c03_3seeds rescue)
#
# 2 conds × 1 seed; expand to seeds 2,3 if cond 0 or 1 lifts above 0.851.
# ============================================================================

CONDS=(
  "Euclidean:0.0:vitl_eucl"
  "PoincareBall:0.3:vitl_poincare_c03"
)

IFS=':' read -r MANIFOLD CURV CND_TAG <<< "${CONDS[$SLURM_ARRAY_TASK_ID]}"

if [ "${MANIFOLD}" = "Euclidean" ]; then
    HC_START=0.0
    HC_END=0.0
else
    HC_START=0.0
    HC_END=5.0
fi

echo "=== inat18_vitl_ours task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} ==="
echo "    backbone=dinov2_vitl14  res=336  manifold=${MANIFOLD}  c=${CURV}  hc_end=${HC_END}"

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
    dataset.resize=336 \
    experiment.seed=1 \
    experiment.batch_size=64 \
    experiment.log_interval=5 \
    experiment.num_workers=6 \
    architecture.model_type=dinov2_vitl14 \
    architecture.latent_dim=128 \
    architecture.projection_layer_dims='[1024,512]' \
    architecture.manifold_name=${MANIFOLD} \
    architecture.curvature_magnitude=${CURV} \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.augmentation=byol \
    pretraining.epochs=30 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=False \
    pretraining.pretrained_weights_path=null \
    pretraining.contrastive_loss.weight=0.5 \
    pretraining.contrastive_loss.temperature=0.2 \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.unfreeze_backbone_from=null \
    clustering.accumulate_gradients=4 \
    clustering.hyperbolic_loss.start_weight=${HC_START} \
    clustering.hyperbolic_loss.end_weight=${HC_END} \
    clustering.hyperbolic_loss.weight_interval=20 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.temperature_interval=20 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    clustering.lr_scheduler.interval=10 \
    +clustering.early_stop.enabled=True \
    +clustering.early_stop.monitor=test/1_nearest_neighbors_accuracy \
    +clustering.early_stop.threshold=0.001 \
    +clustering.early_stop.warmup_epochs=15 \
    +clustering.early_stop.mode=max \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.spectral_clustering=False \
    evaluation_metrics.test_sample_size=0 \
    wandb.tags='[inat18,vitl,ours,frozen,d128,res336,paper_main,seed1,vsc,'${CND_TAG}']'
