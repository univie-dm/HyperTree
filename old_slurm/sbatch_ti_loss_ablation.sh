#!/bin/bash
#SBATCH --job-name=ti_loss_abl
#SBATCH --time=12:00:00
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/ti_loss_abl_%A_%a.log
#SBATCH -A datamining
#SBATCH -x galadriel,dgx-h100-em2,vader
#SBATCH --array=3-5%3

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# TINYIMAGENET LOSS ABLATION (rebuttal AC-1 / Gy9D-W2,Q3).
#
# Gy9D asks for the contrastive-only / hierarchical-only / joint decomposition
# ACROSS benchmarks. CIFAR-100 alone answers it on one dataset, and the result
# there is ambiguous (L_C-only 0.878 vs joint 0.841 on Ward-hDP at use_root
# =True, with the joint arm carrying 3-5x the seed variance of every other
# arm). TinyImageNet d=16 is the right second dataset: it is the other
# configuration verified to use the paper's w = 1/(1+d) objective
# (use_backbone_knn_similarities=False), and it is where our low-dimensional
# result beats supervised HIER by the widest margin.
#
# The joint arm already exists (baselines/rebuild/ti_poin_d16_seed{42,43,44}),
# so this supplies the two missing arms on the identical recipe:
#   cond 0: contrastive-only  (L_H end_weight = 0, contrastive on at 0.1)
#   cond 1: hierarchical-only (L_H end_weight = 5, contrastive off)
#
# Everything else matches the joint arm exactly, including the warm temperature
# schedule TI uses (1.0 -> 0.5) and the cached SimCLR encoder, so the three
# arms share one anchor. ~6h20m per seed (encoder cached; the 20.9h figure in
# rebuttal_data/wallclock.md was a cold run that pretrained in-job).
# ============================================================================

SEEDS=(42 43 44)
COND_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
SEED=${SEEDS[$SEED_IDX]}

case $COND_IDX in
  0) HC_W=0.0; C_EN=True;  C_W=0.1; ARM=conly  ;;
  1) HC_W=5.0; C_EN=False; C_W=0.0; ARM=lhonly ;;
esac
# L_H-only cannot start at weight 0: with contrastive off the total loss would
# be identically zero at epoch 0 and backward fails (no grad_fn). Mirror the
# CIFAR L_H-only recipe, which starts at 1.0.
HL_START=$([ "$ARM" = "lhonly" ] && echo 1.0 || echo 0.0)

export HYP_DUMP_EMB=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/rebuild/ti_${ARM}_d16_seed${SEED}
echo "=== ti_loss_abl task ${SLURM_ARRAY_TASK_ID}: arm=${ARM} HC_W=${HC_W} contrastive=${C_EN}/${C_W} seed=${SEED} ==="

uv run python train.py --config-name tiny_imagenet_simclr \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=512 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=16 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=800 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.augmentation=byol \
    pretraining.contrastive_loss.temperature=0.5 \
    clustering.epochs=600 \
    clustering.freeze_encoder=True \
    clustering.contrastive_loss.enabled=${C_EN} \
    clustering.contrastive_loss.weight=${C_W} \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=${HL_START} \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    wandb.tags='[tiny_imagenet,d16,loss_ablation,'${ARM}',seed'${SEED}']'

module purge
