#!/bin/bash
#SBATCH --job-name=hyperbolic_clustering
#SBATCH --time=0-16:00:00
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/hyperbolic_clustering_test.log

export PYTHON_VERSION=3.11
export ENV_NAME=hyper
export ENV_MODE=permanent

source /etc/profile.d/modules.sh
module load miniforge

# Install remaining packages
conda install tqdm seaborn ipdb scikit-learn threadpoolctl ruff py-spy wandb matplotlib \
                jupyterlab einops scikit-learn-extra \
                hydra-core omegaconf -y


pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install lightning icecream clustpy

export CUBLAS_WORKSPACE_CONFIG=:4096:8
# 728 395 291 18 887 93	108	480	868	134
python train.py dataset=gtsrb architecture.ae_type=feedforward pretraining.epochs=1000 \
    pretraining.overwrite=False clustering.epochs=1200 hyphc.loss_weight=1.0 hyphc.normalized_loss=True \
    hyphc.dist_sum_loss=True \
    experiment.gradient_clip_value=1.0 hyphc.triplet_subsampling_size=150 \
    architecture.latent_dim=4 experiment.optimizer=RiemannianSGD pretraining.weight_decay=1e-4 \
    clustering.weight_decay=1e-4 pretraining.augmentations=False clustering.augmentations=False \
    architecture.curvature_magnitude=0.1 hyphc.distances_type=geodesic clustering.lr_scheduler.type=warmup_cosine \
    clustering.lr_scheduler.interval=20 clustering.learning_rate=1e-2 experiment.batch_size=512 \
    pretraining.learning_rate=1e-3 architecture.spectral_normalization=False \
    architecture.activation_function=swish experiment.log_interval=100 hyphc.distances_version=mobius_direct architecture.manifold_name=PoincareBall\
    architecture.manifold_precision=float64 architecture.precision=float64 \
    experiment.seed=291 \
    experiment.num_workers=15 \
    evaluation.dendrogram_purity_single=True evaluation.dendrogram_purity_complete=True \
    evaluation.dendrogram_purity_average=True evaluation.dendrogram_purity_ward=True \
    evaluation.agglomerative_clustering_single=True evaluation.agglomerative_clustering_complete=True \
    evaluation.agglomerative_clustering_average=True evaluation.agglomerative_clustering_ward=True \
    evaluation.nearest_neighbors_accuracy=True \
    wandb.tags=['gtsrb','poincare','float64','test'] +slurm_id=$SLURM_JOB_ID
module purge
