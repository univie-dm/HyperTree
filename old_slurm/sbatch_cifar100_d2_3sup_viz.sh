#!/bin/bash
#SBATCH --job-name=c100_d2_3sup_viz
#SBATCH --time=18:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/cifar100_d2_3sup_viz_%A_%a.log
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
# CIFAR-100 d=2 Poincaré — 3-superclass / 15-fine-class subset, tuned for
# the qualitative title figure. Three knob changes vs the 5-superclass run:
#
#   1. fewer classes  (3 supers x 5 fine = 15 leaves) so each superclass
#      occupies ~120 deg of angular space instead of 72 deg
#   2. stronger HypHC schedule (sw=2, ew=10, et=0.05) to push leaves to
#      the disk boundary harder so radial-fan layout is honest
#   3. more clustering epochs (500 vs 300) and tighter sampling
#      (eps=0.005) for cleaner late-epoch geometry
#
# Subset = 5 fine classes from each of:
#   fish (1), trees (17), vehicles_1 (18).  Picked for max semantic
# distance so the inferred tree should split cleanly at the root.
# ============================================================================

SEED=1
DUMP_DIR=figures/d2_viz
mkdir -p ${DUMP_DIR}
export HYP_DUMP_EMB=${DUMP_DIR}/c100_d2_3sup_poincare_seed${SEED}

echo "=== c100_d2_3sup_viz: seed=${SEED} dump=${HYP_DUMP_EMB} ==="

uv run python train.py \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=cifar100 \
    dataset.selected_classes=[1,32,67,73,91,47,52,56,59,96,8,13,48,58,90] \
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
    clustering.epochs=500 \
    clustering.learning_rate=0.0001 \
    clustering.freeze_encoder=True \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=2.0 \
    clustering.hyperbolic_loss.end_weight=10.0 \
    clustering.hyperbolic_loss.weight_interval=300 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.05 \
    clustering.hyperbolic_loss.temperature_interval=300 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.005 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=True \
    wandb.tags='[cifar100,poincare_d2_viz,3sup,paper_figure,seed'${SEED}']'

module purge
