import omegaconf
import os
import torch

from hashlib import md5
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from typing import Tuple, Union

from src.datasets.datamodule import (
    ColorDataModule,
    GrayScaleDataModule,
    TabularDataModule,
)
from src.models import (
    HyperbolicAE,
    HyperbolicHCModel,
    HyperbolicResNet,
)


def get_model_path(cfg: omegaconf.DictConfig):
    attr = {
        # Experiment attributes
        "seed": cfg.experiment.seed,
        "batch_size": cfg.experiment.batch_size,
        "optimizer": cfg.experiment.optimizer,
        "gradient_clip_value": cfg.experiment.gradient_clip_value,
        # Pretraining attributes
        "pretrain_epochs": cfg.pretraining.epochs,
        "learning_rate": cfg.pretraining.learning_rate,
        "weight_decay": cfg.pretraining.weight_decay,
        "use_contrastive_loss_pretrain": cfg.pretraining.contrastive_loss.enabled,
        "contrastive_weight_pretrain": cfg.pretraining.contrastive_loss.weight,
        "contrastive_temperature_pretrain": cfg.pretraining.contrastive_loss.temperature,
        "lr_scheduler_pretrain": cfg.pretraining.get("lr_scheduler", None),
        # Model attributes
        "architecture": cfg.architecture,
        # Dataset attributes
        "dataset_name": cfg.dataset.name,
        "selected_classes": cfg.dataset.get("selected_classes", None),
        "dataset_subset": cfg.dataset.get("subset", None),
    }
    if cfg.pretraining.load_supervised_weights:
        attr['supervised_weights_path'] = cfg.pretraining.supervised_weights_path
    return md5(str(attr).encode()).hexdigest()


def get_datamodule(
    cfg: omegaconf.DictConfig, stage: str
) -> Tuple[Union[TabularDataModule, GrayScaleDataModule, ColorDataModule], int, bool]:
    # Determine dataset type and create appropriate datamodule
    if hasattr(cfg.dataset, "type") and cfg.dataset.type == "tabular":
        datamodule = TabularDataModule(
            dataset_name=cfg.dataset.name,
            batch_size=cfg.experiment.batch_size,
            num_workers=cfg.experiment.num_workers,
            train_ratio=cfg.dataset.train_ratio,
            aug_noise_std=cfg.dataset.aug_noise_std,
        )
        input_dim = cfg.dataset.input_dim
        is_image = False
    elif hasattr(cfg.dataset, "channels") and cfg.dataset.channels == 1:
        datamodule = GrayScaleDataModule(
            dataset_name=cfg.dataset.name,
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.experiment.batch_size,
            resize=cfg.dataset.resize,
            subset=cfg.dataset.subset,
            num_workers=cfg.experiment.num_workers,
            selected_classes=cfg.dataset.get("selected_classes", None),
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True
    elif hasattr(cfg.dataset, "channels") and cfg.dataset.channels == 3:
        datamodule = ColorDataModule(
            dataset_name=cfg.dataset.name,
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.experiment.batch_size,
            resize=cfg.dataset.resize,
            subset=cfg.dataset.subset,
            num_workers=cfg.experiment.num_workers,
            selected_classes=cfg.dataset.get("selected_classes", None),
            precision=cfg.architecture.precision,
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True
    else:
        raise ValueError("Invalid dataset configuration")
    datamodule.setup()
    return datamodule, input_dim, is_image


def get_pretrain_model(cfg: omegaconf.DictConfig, input_dim: int) -> Union[HyperbolicAE, HyperbolicResNet]:
    if (os.path.exists(f"data/models/{get_model_path(cfg)}.pt") and not cfg.pretraining.overwrite):
        model = torch.load(f"data/models/{get_model_path(cfg)}.pt", weights_only=False)
        print("Loaded model from {}".format(f"data/models/{get_model_path(cfg)}.pt"))
    elif cfg.architecture.model_type == "feedforward":
        model = HyperbolicAE(
            # Architecture
            input_dim=input_dim,
            hidden_dims=cfg.architecture.feedforward_hidden_dims,
            params_dtype=cfg.architecture.precision,
            activation_function=cfg.architecture.activation_function,
            feature_scaling=cfg.architecture.feature_scaling,
            hyperbolic_encoding=cfg.architecture.hyperbolic_encoding,
            latent_dim=cfg.architecture.latent_dim,
            manifold_name=cfg.architecture.manifold_name,
            manifold_precision=cfg.architecture.manifold_precision,
            curvature_magnitude=cfg.architecture.curvature_magnitude,
            hyperbolic_decoding=cfg.architecture.hyperbolic_decoding,
            # Optimizer
            optimizer_name=cfg.experiment.optimizer,
            learning_rate=cfg.pretraining.learning_rate,
            weight_decay=cfg.pretraining.weight_decay,
            lr_scheduler_params=cfg.pretraining.get("lr_scheduler", None),
            # Evaluation & Logging
            stage="pretrain",
            global_epoch=0,
            log_interval=cfg.experiment.log_interval,
            evaluation_metrics=cfg.evaluation_metrics,
            # (Hyperbolic-) Layer specific parameters
            clamping_factor=cfg.architecture.hyperbolic_clamping,
            smoothing_factor=cfg.architecture.hyperbolic_smoothing,
            hyperboloid_layer_learnable_scale=cfg.architecture.hyperboloid_layer_learnable_scale,
            FHCNN_normalize=cfg.architecture.FHCNN_normalize,
        )
    elif cfg.architecture.model_type in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        model = HyperbolicResNet(
            # Architecture
            model_type=cfg.architecture.model_type,
            params_dtype=cfg.architecture.precision,
            activation_function=cfg.architecture.activation_function,
            feature_scaling=cfg.architecture.feature_scaling,
            hyperbolic_encoding=cfg.architecture.hyperbolic_encoding,
            latent_dim=cfg.architecture.latent_dim,
            manifold_name=cfg.architecture.manifold_name,
            manifold_precision=cfg.architecture.manifold_precision,
            curvature_magnitude=cfg.architecture.curvature_magnitude,
            hyperbolic_decoding=cfg.architecture.hyperbolic_decoding,
            # Optimizer
            optimizer_name=cfg.experiment.optimizer,
            learning_rate=cfg.pretraining.learning_rate,
            weight_decay=cfg.pretraining.weight_decay,
            lr_scheduler_params=cfg.pretraining.get("lr_scheduler", None),
            # Evaluation & Logging
            stage="pretrain",
            global_epoch=0,
            log_interval=cfg.experiment.log_interval,
            evaluation_metrics=cfg.evaluation_metrics,
            # (Hyperbolic-) Layer specific parameters
            clamping_factor=cfg.architecture.hyperbolic_clamping,
            smoothing_factor=cfg.architecture.hyperbolic_smoothing,
            hyperboloid_layer_learnable_scale=cfg.architecture.hyperboloid_layer_learnable_scale,
            FHCNN_normalize=cfg.architecture.FHCNN_normalize,
            # ResNet specific parameters
            projection_layer_dims=cfg.architecture.projection_layer_dims,
            input_size=cfg.dataset.resize,
            contrastive_loss_enabled=cfg.pretraining.contrastive_loss.enabled,
            contrastive_weight=cfg.pretraining.contrastive_loss.weight,
            contrastive_temperature=cfg.pretraining.contrastive_loss.temperature,
            load_supervised_weights=cfg.pretraining.load_supervised_weights,
        )
    else:
        raise ValueError("Invalid model type")
    return model


def get_clustering_model(cfg: omegaconf.DictConfig, pretrain_model: Union[HyperbolicAE, HyperbolicResNet]) -> HyperbolicHCModel:
    if cfg.pretraining.reconstruction_loss.weight == 0.0 and cfg.clustering.reconstruction_loss.weight > 0.0:
        print("Warning: Pretraining was done without reconstruction loss, "
              "but clustering is using a non-zero reconstruction weight. Will be set to 0.0 during clustering.")
        cfg.clustering.reconstruction_loss.weight = 0.0
    model = HyperbolicHCModel(
        model=pretrain_model,
        # Optimizer
        learning_rate=cfg.clustering.learning_rate,
        weight_decay=cfg.clustering.weight_decay,
        lr_scheduler_params=cfg.clustering.lr_scheduler
        if hasattr(cfg.clustering, "lr_scheduler")
        else None,
        # Evaluation & Logging
        stage="clustering",
        global_epoch=cfg.pretraining.epochs,
        log_interval=cfg.experiment.log_interval,
        evaluation_metrics=cfg.evaluation_metrics,
        # Hyperbolic Loss parameters
        hyperbolic_loss_start_weight=cfg.clustering.hyperbolic_loss.start_weight,
        hyperbolic_loss_end_weight=cfg.clustering.hyperbolic_loss.end_weight,
        hyperbolic_loss_weight_interval=cfg.clustering.hyperbolic_loss.weight_interval,
        hyperbolic_loss_temperature=cfg.clustering.hyperbolic_loss.temperature,
        lca_version=cfg.clustering.hyperbolic_loss.lca_version,
        loss_version=cfg.clustering.hyperbolic_loss.loss_version,
        similarity_pruning=cfg.clustering.hyperbolic_loss.similarity_pruning,
        use_importance_sampling=cfg.clustering.hyperbolic_loss.use_importance_sampling,
        sampling_delta=cfg.clustering.hyperbolic_loss.sampling_delta,
        sampling_eps=cfg.clustering.hyperbolic_loss.sampling_eps,
        log_approximation_diff=cfg.clustering.hyperbolic_loss.log_approximation_diff,
        # Contrastive Loss parameters
        contrastive_loss_enabled=cfg.clustering.contrastive_loss.enabled,
        contrastive_temperature=cfg.clustering.contrastive_loss.temperature,
        contrastive_weight=cfg.clustering.contrastive_loss.weight,
        # Reconstruction Loss parameters
        recon_weight=cfg.clustering.reconstruction_loss.weight,
        # Regularizer parameters
        regularize_to_origin_enabled=cfg.clustering.regularizer.enabled,
        regularizer_weight=cfg.clustering.regularizer.weight,
    )
    return model


def get_logger(cfg: omegaconf.DictConfig) -> Union[WandbLogger, CSVLogger]:
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            save_dir=cfg.wandb.save_dir,
            dir=cfg.wandb.save_dir,
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            tags=cfg.wandb.tags,
            log_model=False,
        )
    else:
        logger = CSVLogger(save_dir=cfg.wandb.save_dir, name=cfg.wandb.project)
    return logger
