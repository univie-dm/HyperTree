#!/bin/bash
#SBATCH --job-name=lca_dump_ti
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/lca_dump_ti_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-2%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# TinyImageNet d=128 hyphc_warm runs for embedding dump (LCA wrong-rate analysis).
# Mirrors the paper_main h2h_warm recipe at clustering.epochs=100 (down from
# 600) since the trajectory plateaus by ce~50 and we just need stable trained
# embeddings. Reuses cached SimCLR encoder. 3 seeds × 1 dataset = 3 runs.

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DUMP_DIR=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/lca_wrong_rate
mkdir -p ${DUMP_DIR}
# HYP_DUMP_EMB is a prefix: per-eval saves to ${prefix}_ce${ep}.pt
export HYP_DUMP_EMB=${DUMP_DIR}/ti_d128_seed${SEED}

echo "=== lca_dump_ti task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} dump=${HYP_DUMP_EMB} ==="

uv run python train.py --config-name tiny_imagenet_simclr \
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
    clustering.epochs=100 \
    experiment.log_interval=10 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,h2h_warm,lca_dump,seed'${SEED}']'

module purge
