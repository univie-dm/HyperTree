#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "imnet_d32_revert"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/imnet_d32_revert_%A_%a.err
#SBATCH --output=logs/imnet_d32_revert_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --array=0-2%3

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# ImageNet d=32 hyphc_warm under the CIFAR-100 hist_revert config.
#
# RESULTS.md flags d=32 hyphc_warm as the lone ImageNet overshooter:
# peak hDP_ward=0.666 at ce≈18, crash to 0.444 by ce=50. The earlier
# `sbatch_vsc_imagenet_schedule_fix.sh` (low_weight / flat_temp variants)
# was attacking the wrong knob — on CIFAR-100 d=16 the same peak-then-crash
# pattern was traced to three drifted knobs vs the historical *-16-Ours
# recipe. `sbatch_cifar100_poincare_d16_historical_revert.sh` reverted
# those three and held final-epoch hDP at 0.846 ± 0.029 (n=3).
#
# On ImageNet, two of the three drifts are already set correctly:
#   - freeze_encoder = True               (frozen FB DINO protocol)
#   - sampling_eps    = 0.01              (pinned in configs/imagenet_dino.yaml)
# The remaining knob is use_backbone_knn_similarities, which the d=32
# warm paper_main sbatch had at True. Empirically d=128 warm bk=0
# (n=3, seeds 42/43/44) already runs cleanly at 0.685 ± 0.009 — same
# config minus this one flag — so the fix is expected to transfer.
#
# 1 condition × 3 seeds = 3 runs. 50 ep clustering. Frozen FB DINO
# ResNet50; no pretraining stage (epochs=0 + DINO_CKPT load).
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

LOCAL_DATA="/tmp/imagenet"
if [ ! -d "$LOCAL_DATA/train" ] && [ -d "$DATA/imagenet" ]; then
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/imagenet/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA/train" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="$DATA/imagenet"
fi

SEEDS=(1 2 3)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== imnet_d32_revert task ${SLURM_ARRAY_TASK_ID}: seed=${SEED} hist_revert (bk=False) ==="

uv run python train.py --config-name imagenet_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset=imagenet \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=32 \
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
    clustering.contrastive_loss.enabled=True \
    clustering.contrastive_loss.weight=0.1 \
    clustering.contrastive_loss.temperature=0.5 \
    clustering.hyperbolic_loss.start_weight=1.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=False \
    clustering.hyperbolic_loss.sampling_eps=0.01 \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.test_sample_size=10000 \
    evaluation_metrics.use_wordnet_labels=True \
    evaluation_metrics.imagenet_hierarchy_path='./data/imagenet_wordnet_hierarchy.txt' \
    wandb.tags='[imagenet,hist_revert,d32,h2h_warm,bk0,paper_main,seed'${SEED}',vsc]'
