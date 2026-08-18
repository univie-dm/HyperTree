#!/bin/bash
#SBATCH --job-name=c100_d2_5sup_poincare_viz
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cifar100_d2_5sup_poincare_viz_%A_%a.log
#SBATCH -A datamining
#SBATCH --partition=p_csunivie_gres,p_datamining
#SBATCH -x galadriel,dgx-h100-em2,dgx1
#SBATCH --array=0-0

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# CIFAR-100 d=2 Poincaré — 5-superclass / 25-fine-class subset for the
# qualitative figure. The full 100-class run at d=2 boundary-saturates and
# loses to Euclidean on hDP_ward (0.15 vs 0.27); the 25-leaf tree fits
# comfortably inside the c=1 disk's hyperbolic diameter (~6.6) so the
# geometry should now visibly help.
#
# Subset = 5 fine classes from each of:
#   fish (1), furniture (6), people (14), trees (17), vehicles_1 (18).
# ============================================================================

SEED=1
DUMP_DIR=figures/d2_viz
mkdir -p ${DUMP_DIR}
export HYP_DUMP_EMB=${DUMP_DIR}/c100_d2_5sup_poincare_seed${SEED}

echo "=== c100_d2_5sup_poincare_viz: seed=${SEED} dump=${HYP_DUMP_EMB} ==="

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    dataset.selected_classes=[1,2,5,8,11,13,20,25,32,35,46,47,48,52,56,58,59,67,73,84,90,91,94,96,98] \
    experiment.seed=${SEED} \
    experiment.batch_size=1024 \
    experiment.log_interval=10 \
    experiment.gradient_clip_value=5 \
    experiment.optimizer=Adam \
    architecture.model_type=resnet50 \
    architecture.latent_dim=2 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    architecture.projection_layer_dims=[512,256,128] \
    architecture.hyperbolic_smoothing=50.0 \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.learning_rate=0.003 \
    pretraining.contrastive_loss.enabled=True \
    pretraining.contrastive_loss.weight=1.0 \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=800 \
    clustering.learning_rate=0.0001 \
    clustering.freeze_encoder=True \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.weight_interval=500 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.temperature_interval=500 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,poincare_d2_viz,5sup,paper_figure,seed'${SEED}']'

module purge
