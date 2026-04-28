#!/bin/bash
#ASC --vanilla
#SBATCH --job-name "inat18_diag_a"
#SBATCH --gres=gpu:4
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos zen4_0768_h100x4
#SBATCH --time=1-00:00:00
#SBATCH --error=logs/inat18_diag_a_%A_%a.err
#SBATCH --output=logs/inat18_diag_a_%A_%a.out
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=348G
#SBATCH --array=0-11%8

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((20000 + (SLURM_JOB_ID + SLURM_ARRAY_TASK_ID * 97) % 45000))
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ============================================================================
# iNat18 diagnostic sweep A — head/training axis to fix the c=0.3 boundary
# saturation diagnosed from wandb trajectories on 2026-04-28:
#
# All c=0.3 rescue runs hit norm_max = norm_mean = 1.826 (= 1/sqrt(0.3),
# i.e. on the Poincaré ball boundary surface) from epoch 11 onward, and
# n@1 stays pinned at 0.000164 (= 4/24426) for the entire trajectory while
# hyperbolic_loss decreases monotonically. Loss decoupled from metric;
# head produces degenerate boundary outputs.
#
# This sweep tests four interventions, all on existing FB DINO ResNet50
# features (no new extraction needed). 4 conditions × 3 seeds = 12 runs.
#
#   cond 0 (d256):    larger latent dim — more directional capacity on
#                     the boundary (n-1)-sphere.  arch.latent_dim=256.
#   cond 1 (cw1.0):   stronger contrastive loss — force instance-level
#                     separation. clustering.contrastive_loss.weight=1.0
#                     (default 0.1, 10x).
#   cond 2 (lr1e-5):  10x lower lr — slow head movement, avoid jumping to
#                     the boundary in 1-2 epochs. clustering.learning_rate.
#   cond 3 (Hyp):     Hyperboloid manifold — no boundary, no tanh
#                     saturation. arch.manifold_name=Hyperboloid.
#
# All conditions use c=0.3 (best ce=0 anchor among curvatures we have)
# warm HypHC schedule (et=0.5, ew=5.0), backbone_knn=True, 50 ep.
#
# task_id = seed_idx * 4 + cond_idx
#   seed_idx 0,1,2 -> seeds 1, 2, 3 (match rescue sweep for pairing)
# ============================================================================

DINO_CKPT="data/models/dino_resnet50_pretrain.pth"
if [ ! -f "$DINO_CKPT" ]; then
    echo "Downloading FB DINO ResNet50 checkpoint..."
    mkdir -p data/models
    wget -O "$DINO_CKPT" https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
fi

CONDS=(
  "d256:c03_d256_warm"
  "cw1.0:c03_cw10_warm"
  "lr1e-5:c03_lr1e5_warm"
  "Hyp:c03_hyperboloid_warm"
)
SEEDS=(1 2 3)

SEED_IDX=$((SLURM_ARRAY_TASK_ID / 4))
COND_IDX=$((SLURM_ARRAY_TASK_ID % 4))

IFS=':' read -r CND_KEY CND_TAG <<< "${CONDS[$COND_IDX]}"
SEED=${SEEDS[$SEED_IDX]}

echo "=== inat18_diag_a task ${SLURM_ARRAY_TASK_ID}: ${CND_TAG} seed=${SEED} ==="

# Stage data to local SSD if possible
LOCAL_DATA="/tmp/inat18"
if [ ! -d "$LOCAL_DATA" ] && [ -d "$DATA/inat18" ]; then
    echo "Copying iNat18 to local SSD..."
    mkdir -p "$LOCAL_DATA"
    cp -r "$DATA/inat18/." "$LOCAL_DATA/"
fi
if [ -d "$LOCAL_DATA" ]; then
    DATA_DIR=${LOCAL_DATA}
else
    DATA_DIR="data/inat18"
fi

# Per-condition overrides (default = base config)
EXTRA=""
case ${CND_KEY} in
  d256)
    EXTRA="architecture.latent_dim=256"
    ;;
  cw1.0)
    EXTRA="clustering.contrastive_loss.weight=1.0"
    ;;
  lr1e-5)
    EXTRA="clustering.learning_rate=1e-5"
    ;;
  Hyp)
    EXTRA="architecture.manifold_name=Hyperboloid"
    ;;
esac

uv run python train.py --config-name inat18_dino \
    +slurm_id=${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    dataset.data_dir=${DATA_DIR} \
    experiment.seed=${SEED} \
    experiment.batch_size=256 \
    experiment.log_interval=10 \
    experiment.num_workers=6 \
    architecture.model_type=resnet50 \
    architecture.latent_dim=128 \
    architecture.manifold_name=PoincareBall \
    architecture.curvature_magnitude=0.3 \
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
    clustering.hyperbolic_loss.start_weight=0.0 \
    clustering.hyperbolic_loss.end_weight=5.0 \
    clustering.hyperbolic_loss.start_temperature=1.0 \
    clustering.hyperbolic_loss.end_temperature=0.5 \
    clustering.hyperbolic_loss.use_backbone_knn_similarities=True \
    clustering.hyperbolic_loss.snn_k=10 \
    evaluation_metrics.dendrogrampurity_use_root=False \
    evaluation_metrics.spectral_clustering=False \
    evaluation_metrics.test_sample_size=0 \
    ${EXTRA} \
    wandb.tags='[inat18,h2h,dino_frozen,paper_main,diag_a,vsc,'${CND_TAG}',seed'${SEED}']'
