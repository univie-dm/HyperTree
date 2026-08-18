#!/bin/bash
#SBATCH --job-name=bioclip_abl
#SBATCH --time=8:00:00
#SBATCH -c 32
#SBATCH --mem=500G
#SBATCH --gres=gpu:4
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/bioclip_abl_%A_%a.log
#SBATCH -A datamining
#SBATCH -x vader,galadriel
#SBATCH --array=0-5%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# SHORT BIOCLIP TUNING ABLATION (rebuttal Gy9D-Q1).
#
# The full BioCLIP run (663055) degraded every metric over 50 epochs:
#     hDP    0.346 -> 0.150     flat DP 0.341 -> 0.069
#     1-NN   0.373 -> 0.113     R@10    0.645 -> 0.204
# The recipe was tuned for 2048-d DINO ResNet-50 features (curvature 0.3, warm
# schedule, L_H annealed to 5.0, lr 1e-4). BioCLIP is 512-d CLIP-style with a
# different feature scale, so this may be a tuning failure rather than a
# failure of the method. This grid tests the three most likely causes at 20
# epochs, which is enough to see whether the trajectory turns down at all.
#
#   0: c=0.3, W=1.0, lr=1e-4   L_H over-weighted? (50:1 vs contrastive)
#   1: c=1.0, W=5.0, lr=1e-4   curvature too flat for CLIP feature scale?
#   2: c=0.1, W=5.0, lr=1e-4   curvature too curved?
#   3: c=0.3, W=5.0, lr=1e-5   step size too large for 512-d features?
#   4: bk=True,  everything else as the paper recipe  } matched pair isolating
#   5: bk=False, everything else as the paper recipe  } the SNN substitution
#
# Arms 4 and 5 are the controlled version of an accidental comparison: the
# musica run (bk=True, use_root=True) gave 1-NN 0.0001 while the local run
# (bk=False, use_root=False) gave 0.113 on the same backbone. 1-NN does not
# depend on use_root, so the SNN substitution is the likely cause of the leaf
# collapse the paper attributes to the backbone. These two arms differ ONLY in
# that flag and are evaluated identically.
#
# The control is free: the existing 50-epoch run dumped ce10/ce20, evaluated
# separately (job 664107). Compare each arm's ce20 against that.
#
# 1 seed, 20 epochs, dumps every 10.
# ============================================================================

CONFIGS=(
  "0.3:1.0:0.0001:False:wlow"
  "1.0:5.0:0.0001:False:c1p0"
  "0.1:5.0:0.0001:False:c0p1"
  "0.3:5.0:0.00001:False:lrlow"
  "0.3:5.0:0.0001:True:bk1"
  "0.3:5.0:0.0001:False:bk0"
)
IFS=':' read -r CURV HC_W LR BK TAG <<< "${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
SEED=1

DUMP=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/inat_bioclip_abl
mkdir -p ${DUMP}
export HYP_DUMP_EMB=${DUMP}/bioclip_${TAG}_seed${SEED}

echo "=== bioclip_abl ${TAG}: c=${CURV} W=${HC_W} lr=${LR} bk=${BK} seed=${SEED} ==="

uv run --with open_clip_torch python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=inat18 \
    dataset.data_dir=data/inat18 \
    dataset.resize=224 \
    experiment.seed=${SEED} \
    experiment.num_workers=8 \
    architecture.model_type=bioclip \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.curvature_magnitude=${CURV} \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    clustering.epochs=20 \
    clustering.learning_rate=${LR} \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.lca_version=ours \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=${HC_W} \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=${BK} \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.inat18_hierarchy_path='./data/inat18_taxonomy_hierarchy.txt' \
    wandb.tags="[inat18,rebuttal,gy9d_q1,bioclip_abl,${TAG},seed${SEED}]"

module purge
