#!/bin/bash
#SBATCH --job-name=lh_wsweep
#SBATCH --time=8:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/lh_wsweep_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel,dgx-h100-em2,vader
#SBATCH --array=0-3%4

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# L_H WEIGHT SENSITIVITY SWEEP (rebuttal AC-1 / Gy9D-W2,Q3).
#
# The triad at the paper's setting gives, on CIFAR-100 d=16 (use_root=True):
#     L_C-only (w=0)   0.871
#     joint    (w=5)   0.841
#     L_H-only         0.761
# i.e. L_H at w=5 is net-neutral-to-negative on Ward-hDP. But L_H is clearly
# doing something structural: the joint and L_H-only arms land on IDENTICAL
# sibling-triplet (0.321/0.323) and origin-depth rho (-0.050/-0.050) values
# despite differing in the contrastive term, and both differ sharply from the
# L_C-only arm (0.812 / -0.560). That is the signature of L_H DOMINATING the
# objective, not of it being inert -- note the weight ratio is 50:1 against
# contrastive (w=0.1). The joint arm is also unstable across epochs
# (0.37 -> 0.72) where L_C-only is steady, a classic over-weighting signature.
#
# Hypothesis: L_H is over-weighted at 5.0, not useless. This sweeps the
# annealing endpoint while preserving the paper's 1:5 start:end schedule shape.
# W=5.0 reproduces the joint config exactly (start 1.0 -> end 5.0), so it acts
# as a sanity check against the existing number.
#
# Seed 1 only, 4 configs in parallel, cached encoder (~2.5 h each).
# ============================================================================

W_ENDS=(0.1 0.5 1.0 2.0)
W_END=${W_ENDS[$SLURM_ARRAY_TASK_ID]}
W_START=$(uv run python -c "print(${W_END}/5.0)")
SEED=1
TAG=$(echo ${W_END} | tr '.' 'p')

export HYP_DUMP_EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/rebuild/c100_wsweep_w${TAG}_seed${SEED}
echo "=== lh_wsweep task ${SLURM_ARRAY_TASK_ID}: end_weight=${W_END} start=${W_START} seed=${SEED} ==="

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
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.learning_rate=0.003 \
    pretraining.contrastive_loss.enabled=True \
    pretraining.contrastive_loss.weight=1.0 \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=300 \
    clustering.learning_rate=0.0001 \
    clustering.freeze_encoder=True \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=${W_START} \
    clustering.hyperbolic_loss.end_weight=${W_END} \
    clustering.hyperbolic_loss.weight_interval=250 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.temperature_interval=250 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[cifar100,poincare_d16,lh_wsweep,w'${TAG}',ablation,seed'${SEED}']'

module purge
