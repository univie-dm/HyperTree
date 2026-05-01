#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_vitb"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/inat18_vitb_%A_%a.err
#SBATCH --output=logs/inat18_vitb_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-3%4

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 frozen ViT-B clustering — 10-epoch viability check.
#
# All prior iNat sweeps used a frozen FB DINO ResNet50 backbone; every
# Poincaré variant collapses (n@1 < 0.01) and the Euclidean ceiling is
# n@1 = 0.0103. The diagnosis from the c=0.3 rescue trajectory was
# "backbone information content" — RN50-DINO trained on ImageNet (1000
# classes) hasn't been pushed to discriminate fine-grained species.
#
# This sweep tests whether a stronger frozen ViT-B backbone fixes that:
#   cond 0,1: dino_vitb16   (DINOv1, 768-d, ImageNet SSL pretrain)
#   cond 2,3: dinov2_vitb14 (DINOv2, 768-d, LVD-142M includes iNat-style data)
#
# 50 clustering epochs with eval every 5 epochs (log_interval=5) — wandb
# captures the trajectory, and CollapseKillCallback stops a run early when
# test/1_nearest_neighbors_accuracy stays at/below 0.001 (~10× random for
# 8142 classes) after a 15-epoch warmup, so collapsed Poincaré configs
# don't burn the full 24h budget.
#
# Each cond runs Poincaré c=0.1 (HIER-style flat) and Euclidean for a
# manifold-vs-backbone read.
#
# 4 conds × 1 seed = 4 runs. If any cond shows a clearly rising trajectory
# vs the RN50 baseline, expand to 50 epochs + multiple seeds.
#
# task_id layout:
#   0: dino_vitb16,   Euclidean
#   1: dino_vitb16,   PoincareBall c=0.1
#   2: dinov2_vitb14, Euclidean
#   3: dinov2_vitb14, PoincareBall c=0.1
# ============================================================================

CONDS=(
  "dino_vitb16:Euclidean:0.0:dinov1_vitb16_euclidean"
  "dino_vitb16:PoincareBall:0.1:dinov1_vitb16_poincare_c01"
  "dinov2_vitb14:Euclidean:0.0:dinov2_vitb14_euclidean"
  "dinov2_vitb14:PoincareBall:0.1:dinov2_vitb14_poincare_c01"
)

IFS=':' read -r BACKBONE MANIFOLD CURV CND_TAG <<< "${CONDS[$SLURM_ARRAY_TASK_ID]}"

echo "=== inat18_vitb task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} ==="
echo "    backbone=${BACKBONE}  manifold=${MANIFOLD}  c=${CURV}"

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

# torch.hub fetches DINO weights (~330 MB DINOv1 / ~340 MB DINOv2) on first
# call; subsequent tasks read from cache at ~/.cache/torch/hub.

uv run python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=1 \
    experiment.batch_size=128 \
    experiment.log_interval=5 \
    experiment.num_workers=6 \
    architecture.model_type=${BACKBONE} \
    architecture.latent_dim=128 \
    architecture.projection_layer_dims='[1024,512]' \
    architecture.manifold_name=${MANIFOLD} \
    architecture.curvature_magnitude=${CURV} \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=False \
    pretraining.pretrained_weights_path=null \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.unfreeze_backbone_from=null \
    clustering.accumulate_gradients=2 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
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
    wandb.tags='[inat18,vitb,frozen,d128,viability,50epochs,vsc,'${CND_TAG}']'
