#!/bin/bash
#SBATCH --job-name=lca_curv_sweep
#SBATCH --time=4:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/lca_curv_sweep_%A_%a.log
#SBATCH -A datamining
#SBATCH --array=0-11%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# §4.2.3 LCA wrong-rate curvature sweep on CIFAR-100 d=16.
#
# We already have c=1.0 dumps showing trajectory wrong-rate ≈ 0%. To make
# the rectification ablation meaningful, push curvature to extremes where
# the synthetic norm-band sweep predicts non-trivial wrong-rates: at high
# c the manifold radius (1/√c) shrinks so embeddings either pile near the
# boundary or get squashed near origin.
#
# 4 curvatures × 3 seeds = 12 runs. Pretraining cache is keyed on full
# `cfg.architecture` (utils.py:45) so each curvature gets its own SimCLR
# pretrain (~2h) + 50 ep clustering with embedding dumps (~30 min).
#
# Dumps go to baselines/lca_curvature_sweep/c{CURV}/c100_d16_seed{SEED}_ce{EP}.pt
# Run sbatch_lca_curvature_aggregate.sh after this finishes.

CURVATURES=(0.01 0.1 10.0 100.0)
SEEDS=(1 2 3)
CURV_IDX=$((SLURM_ARRAY_TASK_ID % 4))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 4))
CURV=${CURVATURES[$CURV_IDX]}
SEED=${SEEDS[$SEED_IDX]}

DUMP_DIR=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/lca_curvature_sweep/c${CURV}
mkdir -p ${DUMP_DIR}
export HYP_DUMP_EMB=${DUMP_DIR}/c100_d16_seed${SEED}

echo "=== lca_curv_sweep task ${SLURM_ARRAY_TASK_ID}: c=${CURV} seed=${SEED} dump=${HYP_DUMP_EMB} ==="

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    experiment.seed=${SEED} \
    experiment.batch_size=1024 \
    experiment.log_interval=10 \
    experiment.gradient_clip_value=5 \
    experiment.optimizer=Adam \
    architecture.model_type=resnet50 \
    architecture.latent_dim=16 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    architecture.projection_layer_dims=[512,256,128] \
    architecture.hyperbolic_smoothing=50.0 \
    architecture.curvature_magnitude=${CURV} \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.learning_rate=0.003 \
    pretraining.contrastive_loss.enabled=True \
    pretraining.contrastive_loss.weight=1.0 \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=50 \
    experiment.log_interval=5 \
    clustering.learning_rate=0.0001 \
    clustering.freeze_encoder=True \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.weight_interval=250 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.temperature_interval=250 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,d16,hist_revert,lca_curv_sweep,c'${CURV}',seed'${SEED}']'

module purge
