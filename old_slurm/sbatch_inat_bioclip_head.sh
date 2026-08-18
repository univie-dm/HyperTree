#!/bin/bash
#SBATCH --job-name=inat_bioclip_head
#SBATCH --time=2-00:00:00
#SBATCH -c 48
#SBATCH --mem=500G
#SBATCH --gres=gpu:4
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/inat_bioclip_head_%A_%a.log
#SBATCH -A datamining
#SBATCH -x vader,galadriel
#SBATCH --array=0-2%2

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent
source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# Rebuttal Gy9D-Q1/W1 PHASE 2: train HyperTree's hyperbolic head on a FROZEN
# BioCLIP backbone (biology-domain ViT-B/16) on iNat18.
#
# Phase 1 (job 662777) established the frozen-backbone controls on iNat18 val
# under the pinned protocol:
#     random floor      hDP 0.138   1-NN 0.000
#     DINO ResNet-50    hDP 0.281   1-NN 0.091
#     BioCLIP           hDP 0.552   1-NN 0.496
# That shows the leaf-level collapse is a property of the BACKBONE. It does
# NOT show that HyperTree adds anything on top of an adequate backbone --
# which is what Gy9D actually asks ("if it recovers, it would significantly
# strengthen the scalability claim"). Reporting Phase 1 alone invites the
# reading that the backbone does all the work.
#
# This run supplies the missing arm. Same recipe as the iNat18 headline
# (d=128, PoincareBall, c=0.3, 50 clustering epochs, frozen encoder); only
# the backbone differs. The matched comparison is this run's OWN ce=0 anchor
# (frozen BioCLIP + untrained head) vs its trained endpoint.
#
# NOTE ON THE OBJECTIVE: use_backbone_knn_similarities=False, i.e. the
# w = 1/(1+d) weighting the paper states in Sec 4.1, rather than the frozen
# backbone SNN-Jaccard substitute. Both arms share the same anchor, so the
# delta is internally consistent either way; False is chosen so the number
# corresponds to the method as described in the paper.
#
# Dumps embeddings for offline eval under the pinned protocol.
# ============================================================================

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

DUMP_DIR=/srv/home/users/sidakk95cs/hyperbolic-clustering/baselines/inat_bioclip
mkdir -p ${DUMP_DIR}
export HYP_DUMP_EMB=${DUMP_DIR}/inat_bioclip_seed${SEED}

echo "=== inat_bioclip_head task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run --with open_clip_torch python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=inat18 \
    dataset.data_dir=data/inat18 \
    dataset.resize=224 \
    experiment.seed=${SEED} \
    architecture.model_type=bioclip \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.curvature_magnitude=0.3 \
    architecture.manifold_precision=float64 \
    architecture.precision=float32 \
    pretraining.method=contrastive \
    pretraining.epochs=0 \
    clustering.epochs=50 \
    clustering.freeze_encoder=True \
    clustering.hyperbolic_loss.lca_version=ours \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.inat18_hierarchy_path='./data/inat18_taxonomy_hierarchy.txt' \
    wandb.tags="[inat18,rebuttal,gy9d_q1,bioclip_head,d128,c03,seed${SEED}]"

module purge
