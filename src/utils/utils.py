import omegaconf

from hashlib import md5
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from typing import Union

from src.datasets.datamodule import ColorDataModule
from src.models import HyperTreeModel, HyperbolicResNet
from src.models.hypertree import get_training_views
from src.models.resnet import PRETRAINING_VIEWS


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
        "accumulate_gradients_pretrain": cfg.pretraining.accumulate_gradients,
        "contrastive_temperature_pretrain": cfg.pretraining.contrastive_loss.temperature,
        "lr_scheduler_pretrain": cfg.pretraining.lr_scheduler,
        "pretrained_weights_path": cfg.pretraining.pretrained_weights_path,
        # Model attributes
        "architecture": cfg.architecture,
        # Dataset attributes
        "dataset_name": cfg.dataset.name,
        "selected_classes": cfg.dataset.selected_classes,
        "dataset_subset": cfg.dataset.subset,
    }
    return md5(str(attr).encode()).hexdigest()


def get_datamodule(cfg: omegaconf.DictConfig, stage: str) -> ColorDataModule:
    # The training dataloader only prepares the views that the losses of the stage consume
    if stage == "pretrain":
        train_views = PRETRAINING_VIEWS
    else:
        hierarchical_view, contrastive_views = get_training_views(
            cfg.clustering.hierarchical_loss.view,
            cfg.clustering.contrastive_loss.positive_pair,
            cfg.clustering.contrastive_loss.weight > 0.0 and cfg.clustering.contrastive_loss.enabled,
        )
        train_views = tuple(dict.fromkeys((hierarchical_view, *contrastive_views)))
    datamodule = ColorDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.experiment.batch_size,
        resize=cfg.dataset.resize,
        subset=cfg.dataset.subset,
        num_workers=cfg.experiment.num_workers,
        selected_classes=cfg.dataset.selected_classes,
        precision=cfg.architecture.precision,
        train_views=train_views,
    )
    datamodule.setup()
    return datamodule


def get_pretrain_model(cfg: omegaconf.DictConfig) -> HyperbolicResNet:
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
        # Optimizer
        optimizer_name=cfg.experiment.optimizer,
        learning_rate=cfg.pretraining.learning_rate,
        weight_decay=cfg.pretraining.weight_decay,
        lr_scheduler_params=cfg.pretraining.lr_scheduler,
        # Evaluation & Logging
        stage="pretrain",
        global_epoch=0,
        log_interval=cfg.experiment.log_interval,
        evaluation_metrics=cfg.evaluation_metrics,
        # ResNet specific parameters
        projection_layer_dims=cfg.architecture.projection_layer_dims,
        input_size=cfg.dataset.resize,
        contrastive_temperature=cfg.pretraining.contrastive_loss.temperature,
        pretrained_weights_path=cfg.pretraining.pretrained_weights_path,
    )
    return model


def get_clustering_model(cfg: omegaconf.DictConfig, pretrain_model: HyperbolicResNet) -> HyperTreeModel:
    model = HyperTreeModel(
        model=pretrain_model,
        # Optimizer
        learning_rate=cfg.clustering.learning_rate,
        weight_decay=cfg.clustering.weight_decay,
        lr_scheduler_params=cfg.clustering.lr_scheduler,
        # Evaluation & Logging
        stage="clustering",
        global_epoch=cfg.pretraining.epochs,
        log_interval=cfg.experiment.log_interval,
        evaluation_metrics=cfg.evaluation_metrics,
        # Hierarchical Loss parameters
        hierarchical_loss_start_weight=cfg.clustering.hierarchical_loss.start_weight,
        hierarchical_loss_end_weight=cfg.clustering.hierarchical_loss.end_weight,
        hierarchical_loss_weight_interval=cfg.clustering.hierarchical_loss.weight_interval,
        hierarchical_loss_start_temperature=cfg.clustering.hierarchical_loss.start_temperature,
        hierarchical_loss_end_temperature=cfg.clustering.hierarchical_loss.end_temperature,
        hierarchical_loss_temperature_interval=cfg.clustering.hierarchical_loss.temperature_interval,
        hierarchical_loss_view=cfg.clustering.hierarchical_loss.view,
        lca_version=cfg.clustering.hierarchical_loss.lca_version,
        sampling_eps=cfg.clustering.hierarchical_loss.sampling_eps,
        sampling_delta=cfg.clustering.hierarchical_loss.sampling_delta,
        # Backbone kNN SNN similarity parameters
        use_backbone_knn_similarities=cfg.clustering.hierarchical_loss.use_backbone_knn_similarities,
        snn_k=cfg.clustering.hierarchical_loss.snn_k,
        # Contrastive Loss parameters
        contrastive_loss_enabled=cfg.clustering.contrastive_loss.enabled,
        contrastive_temperature=cfg.clustering.contrastive_loss.temperature,
        contrastive_weight=cfg.clustering.contrastive_loss.weight,
        contrastive_positive_pair=cfg.clustering.contrastive_loss.positive_pair,
        # Encoder
        freeze_encoder=cfg.clustering.freeze_encoder,
        train_projection_head=cfg.clustering.train_projection_head,
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
