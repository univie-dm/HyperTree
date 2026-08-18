#!/bin/bash
#SBATCH --job-name=ti_ch_smk2
#SBATCH --time=0-1:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/ti_coneheads_smk2_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-0

# Bypass the module/permanent-env wrapper (it wedges rebuilding /venvs/hyper);
# use the repo's prebuilt uv .venv directly.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
export PYTHONPATH=src
PY=.venv/bin/python

# ============================================================================
# TinyImageNet d=128: Coneheads umbral shadow-cone sup2 join AS the HypHC
# training-LCA, vs our ball-geodesic LCA control. Same warm recipe as the
# h2h_warm winner (st=1.0/et=0.5, ew=5, frozen encoder, cached SimCLR pretrain).
# Measures dendrogram purity (hDP_ward, use_root=False) when the loss uses the
# Coneheads join instead of Chami's geodesic LCA.
#
# task_idx = seed_idx*2 + cond_idx
#   seed_idx 0,1,2 = seeds 42, 43, 44
#   cond_idx 0 = lca_version=coneheads
#   cond_idx 1 = lca_version=ours (matched control)
# ============================================================================

SEEDS=(42 43 44)
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEED=${SEEDS[$SEED_IDX]}

case $COND_IDX in
  0) LCA=coneheads; CND_TAG=coneheads_lca ;;
  1) LCA=ours;      CND_TAG=ours_lca      ;;
esac

echo "=== ti_coneheads_lca task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} lca=${LCA} ==="

$PY train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=512 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=2 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.lca_version=${LCA} \
    clustering.hyperbolic_loss.coneheads_radius=0.05 \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.mode=disabled
