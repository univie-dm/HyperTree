import omegaconf
import os
import torch

from hashlib import md5
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from typing import Tuple, Union

from src.datasets.datamodule import ColorDataModule, GrayScaleDataModule, TabularDataModule
from src.models import HyperbolicAE, HyperbolicHCModel, HyperbolicResNet


def get_model_path(cfg: omegaconf.DictConfig):
    attr = {
        # Experiment attributes
        "seed" : cfg.experiment.seed,
        "batch_size" : cfg.experiment.batch_size,
        "optimizer" : cfg.experiment.optimizer,
        # Pretraining attributes
        "pretrain_epochs" : cfg.pretraining.epochs,
        "learning_rate" : cfg.pretraining.learning_rate,
        "pretrain_trainable_curvature" : cfg.pretraining.trainable_curvature,
        # Model attributes
        "ae_type" : cfg.architecture.ae_type,
        "hidden_dims" : cfg.architecture.hidden_dims,
        "latent_dim" : cfg.architecture.latent_dim,
        "hyperbolic_encoding" : cfg.architecture.hyperbolic_encoding,
        "hyperbolic_decoding" : cfg.architecture.hyperbolic_decoding,
        "precision" : cfg.architecture.precision,
        "manifold_name" : cfg.architecture.manifold_name,
        "manifold_precision" : cfg.architecture.manifold_precision,
        "curvature_magnitude" : cfg.architecture.curvature_magnitude,
        # Dataset specific attributes
        "dataset_name" : cfg.dataset.name,
        "selected_classes" : cfg.dataset.get('selected_classes', None),
        "dataset_subset" : cfg.dataset.get('subset', None),
    }
    return md5(str(attr).encode()).hexdigest()

def get_datamodule(cfg: omegaconf.DictConfig) -> Tuple[Union[TabularDataModule, GrayScaleDataModule, ColorDataModule], int, bool]:
    # Determine dataset type and create appropriate datamodule
    if hasattr(cfg.dataset, 'type') and cfg.dataset.type == 'tabular':
        datamodule = TabularDataModule(
            dataset_name=cfg.dataset.name,
            batch_size=cfg.experiment.batch_size,
            num_workers=cfg.experiment.num_workers,
            train_ratio=cfg.dataset.train_ratio,
            aug_noise_std=cfg.dataset.aug_noise_std
        )
        input_dim = cfg.dataset.input_dim
        is_image = False
    elif hasattr(cfg.dataset, 'channels') and cfg.dataset.channels == 1:
        datamodule = GrayScaleDataModule(
            dataset_name=cfg.dataset.name,
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.experiment.batch_size,
            resize=cfg.dataset.resize,
            subset=cfg.dataset.subset,
            num_workers=cfg.experiment.num_workers,
            selected_classes=cfg.dataset.get('selected_classes', None),
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True
    elif hasattr(cfg.dataset, 'channels') and cfg.dataset.channels == 3:
        datamodule = ColorDataModule(
            dataset_name=cfg.dataset.name,
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.experiment.batch_size,
            resize=cfg.dataset.resize,
            subset=cfg.dataset.subset,
            num_workers=cfg.experiment.num_workers,
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True
    else:
        raise ValueError("Invalid dataset configuration")
    datamodule.setup()
    return datamodule, input_dim, is_image

def get_pretrain_model(cfg: omegaconf.DictConfig, input_dim: int) -> Union[HyperbolicAE, HyperbolicResNet]:
    if os.path.exists(f"data/models/{get_model_path(cfg)}.pt") and not cfg.pretraining.overwrite:
        model = torch.load(f"data/models/{get_model_path(cfg)}.pt", weights_only=False)
        print("Loaded model from {}".format(f"data/models/{get_model_path(cfg)}.pt"))
    elif cfg.architecture.ae_type == 'feedforward':
        model = HyperbolicAE( 
            input_dim=input_dim,
            hidden_dims=cfg.architecture.hidden_dims,
            hyperbolic_encoding=cfg.architecture.hyperbolic_encoding,
            params_dtype=cfg.architecture.precision,
            latent_dim=cfg.architecture.latent_dim,
            manifold_name=cfg.architecture.manifold_name,
            manifold_precision=cfg.architecture.manifold_precision,
            curvature_magnitude=cfg.architecture.curvature_magnitude,
            trainable_curvature=cfg.pretraining.trainable_curvature,
            hyperbolic_decoding=cfg.architecture.hyperbolic_decoding,
            optimizer_name=cfg.experiment.optimizer,
            learning_rate=cfg.pretraining.learning_rate,
            log_interval=cfg.evaluation.log_interval,
            stage='pretrain',
            global_epoch=0,
        )
    elif cfg.architecture.ae_type in ['resnet18', 'resnet34', 'resnet50']:
        model = HyperbolicResNet(
            model_type=cfg.architecture.ae_type,
            input_channels=cfg.dataset.channels,
            hyperbolic_encoding=cfg.architecture.hyperbolic_encoding,
            params_dtype=cfg.architecture.precision,
            latent_dim=cfg.architecture.latent_dim,
            manifold_name=cfg.architecture.manifold_name,
            manifold_precision=cfg.architecture.manifold_precision,
            curvature_magnitude=cfg.architecture.curvature_magnitude,
            trainable_curvature=cfg.pretraining.trainable_curvature,
            hyperbolic_decoding=cfg.architecture.hyperbolic_decoding,
            output_size=cfg.dataset.resize,
            optimizer_name=cfg.experiment.optimizer,
            learning_rate=cfg.pretraining.learning_rate,
            log_interval=cfg.evalutation.log_interval,
        )
    else:
        raise ValueError("Invalid autoencoder type")
    return model

def get_clustering_model(cfg: omegaconf.DictConfig,
                         pretrain_model:  Union[HyperbolicAE, HyperbolicResNet],
                         use_contrastive_loss: bool = False,
                         contrastive_weight: float = 0.5,
                         contrastive_temperature: float = 0.05,
                         ) -> Union[ HyperbolicHCModel]:
    model = HyperbolicHCModel(
        ae_model=pretrain_model,
        clustering_trainable_curvature=cfg.clustering.trainable_curvature,
        learning_rate=cfg.clustering.learning_rate,
        hyphc_weight=cfg.hyphc.loss_weight,
        hyphc_temperature=cfg.hyphc.temperature,
        hyphc_sampling_strategy=cfg.hyphc.sampling_strategy,
        hyphc_triplet_subsampling_size=cfg.hyphc.triplet_subsampling_size,
        lca_version=cfg.hyphc.lca_version,
        start_epoch=cfg.pretraining.epochs,
        log_interval=cfg.evaluation.log_interval,
        stage='clustering',
        global_epoch=cfg.pretraining.epochs,
        learning_rate_scheduler=cfg.clustering.lr_scheduler,
        lr_scheduler_params=cfg.lr_scheduler if hasattr(cfg, 'lr_scheduler') else None,
        learnable_hyphc_weight=cfg.hyphc.learnable_hyphc_weight,
        freeze_similarities=cfg.hyphc.freeze_similarities,
        normalized_hyphc_loss=cfg.hyphc.normalized_loss,
        clustering_adam_momentum=cfg.clustering.adam_momentum,
        hyperbolic_dist_sum_loss=cfg.hyphc.dist_sum_loss,
        use_contrastive_loss=False,  # Contrastive loss is not used in HyperbolicHCModel
        contrastive_weight=0.0,
        contrastive_temperature=0.0,
        distances_type=cfg.hyphc.distances_type,
        evaluation_metrics=cfg.hyphc.evaluation_metrics,
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
            log_model=False
        )
    else:
        logger = CSVLogger(save_dir=cfg.wandb.save_dir, name=cfg.wandb.project)
    return logger
