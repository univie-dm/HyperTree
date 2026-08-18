#!/bin/bash
#SBATCH --job-name=lh_warmstart
#SBATCH --time=8:00:00
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/lh_warmstart_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel,dgx-h100-em2,vader
#SBATCH --array=0-1%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# WARM-START L_H ABLATION (rebuttal AC-1 / Gy9D-W2,Q3).
#
# Why this exists. In the current code the hyperbolic head is NOT trained
# during pretraining (resnet.py: the unsupervised branch computes the
# contrastive loss on backbone features and never calls the hyperbolic layer),
# so it reaches the clustering stage randomly initialised. That means the
# contrastive-only arm gets credit for training the head from scratch, work
# that is not what the paper's claims are about. It also explains why the
# L_C-only arm scores so highly (0.871 vs joint 0.841 on Ward-hDP).
#
# Before commit e6f7c05 (2026-04-12) pretraining DID train the head, via a
# geodesic NT-Xent term on z. In that regime the head arrived converged and
# contrastive had little left to add, which matches the long-standing
# observation that dropping the hyperbolic weight to ~1e-5 left DP flat.
#
# This run poses the paper's actual question without an 800-epoch retrain:
# hold L_H at zero for the first 150 epochs (head_warmup_epochs) so the head
# converges under contrastive alone, then switch L_H on at constant weight for
# the remaining 150. The 300-epoch pure-contrastive run (c100_conly_*) is the
# matched control: both share the same head state at epoch 150, so any
# difference at epoch 300 isolates what L_H contributes ON TOP OF a converged
# head.
#
# Arms: W in {1.0, 5.0}. W=5.0 is the paper's annealing endpoint; W=1.0 tests
# whether the 50:1 ratio against the contrastive weight (0.1) is the issue.
# Seed 1, cached encoder (~2.5 h each).
# ============================================================================

W_ENDS=(1.0 5.0)
W=${W_ENDS[$SLURM_ARRAY_TASK_ID]}
WARMUP=150
SEED=1
TAG=$(echo ${W} | tr '.' 'p')

export HYP_DUMP_EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/rebuild/c100_warmstart_w${TAG}_seed${SEED}
echo "=== lh_warmstart task ${SLURM_ARRAY_TASK_ID}: warmup=${WARMUP} W=${W} seed=${SEED} ==="

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
    +clustering.head_warmup_epochs=${WARMUP} \
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=${W} \
    clustering.hyperbolic_loss.end_weight=${W} \
    clustering.hyperbolic_loss.weight_interval=0 \
    clustering.hyperbolic_loss.start_temperature=0.2 \
    clustering.hyperbolic_loss.end_temperature=0.1 \
    clustering.hyperbolic_loss.temperature_interval=250 \
    clustering.hyperbolic_loss.sampling_alpha=0.2 \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.regularizer.enabled=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[cifar100,poincare_d16,lh_warmstart,w'${TAG}',warmup'${WARMUP}',ablation,seed'${SEED}']'

module purge
