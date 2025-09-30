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
    experiment.seed=$1 \
    experiment.num_workers=15 \
    wandb.tags=['gtsrb','poincare','float64','final2']
module purge
