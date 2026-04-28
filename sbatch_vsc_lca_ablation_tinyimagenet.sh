#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "lca_abl_ti"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/lca_abl_ti_%A_%a.err
#SBATCH --output=logs/lca_abl_ti_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-5%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# LCA-rectification ablation on TinyImageNet — paper Sec 4.2.3.
#
# CIFAR-100 showed Δ < 0.01 on hDP_ward between `lca_version=ours` (rectified
# LCA) and `lca_version=chami`, which doesn't motivate the section. Need at
# least one dataset where ours beats chami clearly. TinyImageNet is the
# strongest hyperbolic story (+0.16 hDP over Euclidean, +0.12 over no-HypHC),
# so it's the dataset where any LCA effect is most likely to show.
#
# Uses the H2H-warm winner recipe (PoincaréBall + HypHC warm, 800 ep SimCLR
# + 600 ep clustering, st=1.0/et=0.5, ew=5). Only `lca_version` varies.
# 2 LCA × 3 seeds = 6 runs.
#
# Array decoding: task = seed_idx * 2 + lca_idx
# ============================================================================

LCAS=("ours" "chami")
SEEDS=(42 43 44)

LCA_IDX=$((SLURM_ARRAY_TASK_ID % 2))
SEED_IDX=$((SLURM_ARRAY_TASK_ID / 2))

LCA=${LCAS[$LCA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== lca_abl_ti task ${SLURM_ARRAY_TASK_ID}: lca=${LCA} seed=${SEED} ==="

uv run python train.py --config-name tiny_imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=${SEED} \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.manifold_precision=float64 \
    clustering.epochs=600 \
    clustering.hyperbolic_loss.lca_version=${LCA} \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.use_wordnet_labels=True \
    wandb.tags='[tiny_imagenet,lca_ablation,d128,h2h_warm,lca_'${LCA}',seed'${SEED}',vsc]'
