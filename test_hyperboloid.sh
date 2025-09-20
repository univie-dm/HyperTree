python train.py dataset=mnist architecture.ae_type=feedforward pretraining.epochs=200 \
    pretraining.overwrite=True clustering.epochs=1000 hyphc.loss_weight=1 hyphc.normalized_loss=True \
    hyphc.dist_sum_loss=True \
    experiment.gradient_clip_value=1 hyphc.triplet_subsampling_size=100 \
    architecture.latent_dim=4 experiment.optimizer=RiemannianSGD pretraining.weight_decay=1e-4 \
    clustering.weight_decay=1e-4 pretraining.augmentations=False clustering.augmentations=False \
    architecture.curvature_magnitude=0.1 hyphc.distances_type=geodesic clustering.lr_scheduler.type=warmup_cosine \
    clustering.lr_scheduler.interval=20 clustering.learning_rate=1e-2 experiment.batch_size=512 \
    pretraining.learning_rate=1e-3 \
    architecture.activation_function=swish experiment.log_interval=10 hyphc.distances_version=default \
    wandb.tags=['mnist','temp']