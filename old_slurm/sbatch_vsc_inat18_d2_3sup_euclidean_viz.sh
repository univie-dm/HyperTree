#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat_d2_3sup_eucl_viz"
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=03:00:00
#SBATCH --error=logs/inat18_d2_3sup_eucl_viz_%j.err
#SBATCH --output=logs/inat18_d2_3sup_eucl_viz_%j.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat-2018 d=2 Euclidean — 3-supercategory / 15-species subset for the
# qualitative compare figure (Euclidean panel, paired with the Poincaré
# sbatch). Uses frozen DINOv2 ViT-L/14 (best iNat features per the
# 2026-04-28 frozen-baseline result) + a thin head trained to d=2.
#
# Selection: 5 species each from Mammalia / Aves / Insecta, picked
# evenly-spaced after sorting by (order, family, genus, id) so the
# 5 leaves inside each phylum span different sub-taxa.
#   class ids = 261,627,1034,1369,2169,3160,3404,3760,3861,3954,
#               4059,4086,4152,4155,4260
#
# 3-supercategory subset → ~3000 train / ~45 val images. Frozen backbone +
# 2-d head → fits in <1 h on a single H100.
# ============================================================================

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

SEED=1
DUMP_DIR=figures/d2_viz
mkdir -p ${DUMP_DIR}
export HYP_DUMP_EMB=${DUMP_DIR}/inat18_d2_3sup_euclidean_seed${SEED}

echo "=== inat18_d2_3sup_eucl_viz: seed=${SEED} dump=${HYP_DUMP_EMB} ==="

uv run python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID} \
    dataset.data_dir=${DATA_DIR} \
    dataset.selected_classes=[261,627,1034,1369,2169,3160,3404,3760,3861,3954,4059,4086,4152,4155,4260] \
    experiment.seed=${SEED} \
    experiment.batch_size=128 \
    experiment.log_interval=10 \
    experiment.num_workers=12 \
    experiment.gradient_clip_value=5 \
    architecture.model_type=dinov2_vitl14 \
    architecture.latent_dim=2 \
    architecture.manifold_name=Euclidean \
    architecture.curvature_magnitude=0.0 \
    architecture.projection_layer_dims='[1024,512,128]' \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=False \
    pretraining.pretrained_weights_path=null \
    clustering.epochs=300 \
    clustering.learning_rate=1e-4 \
    clustering.freeze_encoder=True \
    clustering.unfreeze_backbone_from=null \
    clustering.accumulate_gradients=1 \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=0.0 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    clustering.enable_checkpointing=True \
    evaluation_metrics.dendrogrampurity_use_root=True \
    evaluation_metrics.test_sample_size=0 \
    evaluation_metrics.spectral_clustering=False \
    wandb.tags='[inat18,euclidean_d2_viz,3sup,paper_figure,dinov2_vitl14,seed'${SEED}']'
