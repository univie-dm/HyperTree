#!/bin/bash
#SBATCH --job-name=ti_ch_rsw
#SBATCH --time=0-16:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/ti_ch_rsweep_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel
#SBATCH --array=0-8

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_MODE=offline
cd /srv/home/users/sidakk95cs/hyperbolic-clustering
export PYTHONPATH=src
PY=.venv/bin/python

# Coneheads-LCA radius sweep on TinyImageNet d128. Same recipe/backbone as
# job 657274 (which did r=0.05 + ours-control). New radii to bracket it.
#   task = radius_idx*3 + seed_idx
#   radii = 0.01 (lateral-dominated), 0.2, 1.0 (height-weighted)
#   seeds = 42, 43, 44
RADII=(0.01 0.2 1.0)
SEEDS=(42 43 44)
RIDX=$((SLURM_ARRAY_TASK_ID / 3))
SIDX=$((SLURM_ARRAY_TASK_ID % 3))
RADIUS=${RADII[$RIDX]}
SEED=${SEEDS[$SIDX]}
RTAG=$(echo $RADIUS | tr '.' 'p')

echo "=== ti_ch_rsweep task ${SLURM_ARRAY_TASK_ID}: radius=${RADIUS} seed=${SEED} ==="

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
    +pretraining.pretrained_model_path=data/models/524a4a100c4b1b9f9e3e9b4df2a924bb.pt \
    clustering.epochs=600 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.lca_version=coneheads \
    clustering.hyperbolic_loss.coneheads_radius=${RADIUS} \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.results_dir=./results/ch_rsweep/r${RTAG}_s${SEED} \
    wandb.enabled=True
