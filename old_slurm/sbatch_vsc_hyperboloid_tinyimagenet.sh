#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "hyp_ti"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/hyp_ti_%A_%a.err
#SBATCH --output=logs/hyp_ti_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-2%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# TinyImageNet Hyperboloid manifold ablation. Pairs with the existing
# PoincaréBall + HypHC warm row in §5.4 to support the manifold-agnostic
# claim. Only `manifold_name` differs from `sbatch_vsc_lca_ablation_tinyimagenet.sh`'s
# winner recipe (lca_version=ours, the default for the main row).
#
# Recipe: H2H-warm winner — SimCLR 800ep + 600ep clustering, d=128,
# HypHC warm (st=1.0, et=0.5, ew=5), bk=True.
# 1 condition × 3 seeds = 3 runs.
# ============================================================================

SEEDS=(42 43 44)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== hyp_ti task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} ==="

uv run python train.py --config-name tiny_imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=tiny_imagenet \
    experiment.seed=${SEED} \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=Hyperboloid \
    architecture.hyperbolic_encoding=HyperbolicLinearPP \
    architecture.manifold_precision=float64 \
    clustering.epochs=600 \
    clustering.hyperbolic_loss.lca_version=ours \
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.use_wordnet_labels=True \
    wandb.tags='[tiny_imagenet,manifold_ablation,hyperboloid,d128,h2h_warm,seed'${SEED}',vsc]'
