#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_h2h"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=3-00:00:00
#SBATCH --error=logs/inat18_h2h_%A_%a.err
#SBATCH --output=logs/inat18_h2h_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-3%2

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNaturalist-2018 h2h: hyperbolic vs Euclidean on a dataset with a genuine
# 7-level WordNet-style taxonomy (kingdom > phylum > class > order > family >
# genus > species, 8142 species). This is the flagship "why hyperbolic helps
# on hierarchies" dataset for the paper.
#
# Runs on Vienna Scientific Cluster (zen4_0768_h100x4 partition, 4xH100).
# Pretraining: DINO, 100 epochs, config from configs/inat18_dino.yaml.
# Expect *heavy* pretraining cost: first condition will compute the pretrain
# and cache it; subsequent conditions on the same manifold can reuse via
# pretraining.overwrite=False + matching hash.
#
# 4 conditions (1 seed each for cost). If condition 0 and 2 succeed we have
# the core h2h story. Add seeds later if pretrain cache is warm.
#   0  PoincareBall + HypHC(ew=5), cold schedule (st=0.2, et=0.1)  poincare_hyphc_cold
#   1  PoincareBall + HypHC(ew=5), warm schedule (st=1.0, et=0.5)  poincare_hyphc_warm
#   2  PoincareBall + HypHC(ew=0)                                   poincare_no_hyphc
#   3  Euclidean    + HypHC(ew=0)                                   euclidean_baseline
#
# Note: dataset assumed staged at $DATA/inat18. Override dataset.data_dir below
# if VSC stages data elsewhere.
# ============================================================================

case $SLURM_ARRAY_TASK_ID in
  0) MANIFOLD=PoincareBall; HC_W=5.0; ST_T=0.2; ET_T=0.1; CND_TAG=poincare_hyphc_cold ;;
  1) MANIFOLD=PoincareBall; HC_W=5.0; ST_T=1.0; ET_T=0.5; CND_TAG=poincare_hyphc_warm ;;
  2) MANIFOLD=PoincareBall; HC_W=0.0; ST_T=0.2; ET_T=0.1; CND_TAG=poincare_no_hyphc   ;;
  3) MANIFOLD=Euclidean;    HC_W=0.0; ST_T=0.2; ET_T=0.1; CND_TAG=euclidean_baseline  ;;
esac

echo "=== inat18_h2h task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} manifold=${MANIFOLD} hc_w=${HC_W} st=${ST_T} et=${ET_T} ==="

# Stage data to local SSD if available (edit path for your VSC setup)
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
    experiment.seed=42 \
    architecture.manifold_name=${MANIFOLD} \
    architecture.latent_dim=128 \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=dino \
    pretraining.epochs=100 \
    pretraining.overwrite=False \
    pretraining.accumulate_gradients=4 \
    pretraining.enable_checkpointing=True \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.accumulate_gradients=4 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=${ST_T} \
    clustering.hyperbolic_loss.end_temperature=${ET_T} \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    wandb.tags='[inat18,h2h,vsc,'${CND_TAG}']'

# Clean up local stage (optional)
# rm -rf "$LOCAL_DATA"
