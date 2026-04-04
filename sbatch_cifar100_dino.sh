#!/bin/bash
#SBATCH --job-name=cifar100_dino
#SBATCH --time=2-10:00:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --constraint='nvidia-h100|nvidia-h200'
#SBATCH --output=logs/cifar100_dino_%j.log
#SBATCH -A datamining
#SBATCH -x galadriel

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

export CUBLAS_WORKSPACE_CONFIG=:4096:8

python train.py +slurm_id=${SLURM_JOB_ID} dataset=cifar100 \
    experiment.seed=77 experiment.batch_size=128 experiment.log_interval=10 \
    pretraining.accumulate_gradients=4 \
    architecture.manifold_precision=float64 architecture.precision=float32 \
    architecture.latent_dim=64 \
    pretraining.method=dino pretraining.augmentation=dino \
    pretraining.epochs=600 pretraining.overwrite=True \
    pretraining.learning_rate=5e-4 pretraining.weight_decay=0.04 \
    pretraining.contrastive_loss.temperature=0.1 \
    pretraining.lr_scheduler.interval=60 \
    pretraining.supervised=False \
    pretraining.n_local_crops=4 \
    pretraining.local_crop_size=32 \
    'pretraining.dino_global_scale=[0.7,1.0]' \
    'pretraining.dino_local_scale=[0.3,0.7]' \
    clustering.epochs=1000 \
    evaluation_metrics.dendrogrampurity_use_root=False

module purge
