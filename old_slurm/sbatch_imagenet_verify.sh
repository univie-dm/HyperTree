#!/bin/bash
#SBATCH --job-name=imnet_verify
#SBATCH --time=2-00:00:00
#SBATCH -c 32
#SBATCH --mem=300G
#SBATCH --gres=gpu:4
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/imnet_verify_%A_%a.log
#SBATCH -A datamining
#SBATCH --exclude=vader,galadriel
#SBATCH --array=0-1%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# IMAGENET-1k HIERARCHICAL-ONLY ARM (rebuttal AC-1).
#
# AC-1 names three arms: contrastive-only, hierarchical-only, and joint.
# Table 8 already supplies two of the three on ImageNet-1k at d=128:
#     Euclidean head, contrastive only            0.494
#     Poincare Ball, no L_H (contrastive only)    0.632
#     Poincare Ball + L_H (warm)                  0.685
#     Poincare Ball + L_H (warm, annealed)        0.789
# The hierarchical-only arm (contrastive switched off) is missing. This run
# supplies it, matched to the WARM row rather than the annealed one so that the
# triad sits at a single shared recipe: same manifold, same temperature
# schedule (1.0 -> 0.5), same 50 epochs, same frozen DINO backbone, same
# w = 1/(1+d) objective (use_backbone_knn_similarities=False). The only change
# from that row is clustering.contrastive_loss disabled.
#
# Expected outcome, for reference: on CIFAR-100 the hierarchical-only arm sits
# at its epoch-0 anchor with collapsed retrieval, consistent with the
# boundary-saturation argument in Appendix A.6. Confirming that on ImageNet-1k
# completes the triad on the dataset AC-1 is answered with, and supports the
# paper's stated reason for retaining the contrastive term throughout.
#
# 3 seeds, 2 concurrent, 4x H100 (~13.5 h per seed per wallclock.md).
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

SEED=42
CONDS=("0.0:True:0.1:no_hyphc" "5.0:True:0.1:hyphc_warm")
IFS=":" read -r HC_W C_EN C_W CND <<< "${CONDS[$SLURM_ARRAY_TASK_ID]}"

DUMP_DIR=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/rebuild
mkdir -p ${DUMP_DIR}
export HYP_DUMP_EMB=${DUMP_DIR}/in1k_verify_${CND}_d128_seed${SEED}

echo "=== imnet_verify seed=${SEED} (warm recipe, contrastive OFF) ==="

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    pretraining.overwrite=False \
    pretraining.supervised=False \
    pretraining.load_supervised_weights=False \
    pretraining.pretrained_weights_path=${DINO_CKPT} \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.accumulate_gradients=2 \
    clustering.contrastive_loss.enabled=${C_EN} \
    clustering.contrastive_loss.weight=${C_W} \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,h2h,dino_frozen,d128,'${CND}',bk0,rebuttal,verify,seed'${SEED}']'

module purge
