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
    DINOHyperbolicResNet,
)


def get_model_path(cfg: omegaconf.DictConfig):
    attr = {
        # Experiment attributes
        "seed": cfg.experiment.seed,
        "batch_size": cfg.experiment.batch_size,
        "optimizer": cfg.experiment.optimizer,
        "gradient_clip_value": cfg.experiment.gradient_clip_value,
        # Pretraining attributes
        "pretrain_method": cfg.pretraining.get("method", "contrastive"),
        "pretrain_augmentation": cfg.pretraining.get("augmentation", None),
        "pretrain_epochs": cfg.pretraining.epochs,
        "learning_rate": cfg.pretraining.learning_rate,
        "weight_decay": cfg.pretraining.weight_decay,
        "accumulate_gradients_pretrain": cfg.pretraining.get("accumulate_gradients", 1),
        "use_contrastive_loss_pretrain": cfg.pretraining.contrastive_loss.enabled,
        "contrastive_weight_pretrain": cfg.pretraining.contrastive_loss.weight,
        "contrastive_temperature_pretrain": cfg.pretraining.contrastive_loss.temperature,
        "lr_scheduler_pretrain": cfg.pretraining.get("lr_scheduler", None),
        # Multi-crop attributes (DINO and similar)
        "n_local_crops": cfg.pretraining.get("n_local_crops", None),
        "local_crop_size": cfg.pretraining.get("local_crop_size", None),
        "dino_params": cfg.pretraining.get("dino", None),
        # Model attributes
        "architecture": cfg.architecture,
        # Dataset attributes
        "dataset_name": cfg.dataset.name,
        "selected_classes": cfg.dataset.get("selected_classes", None),
        "dataset_subset": cfg.dataset.get("subset", None),
        "load_supervised_weights": cfg.pretraining.load_supervised_weights,
        "supervised": cfg.pretraining.supervised,
        "pretrained_weights_path": cfg.pretraining.get("pretrained_weights_path", None),
    }
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
        augmentation = cfg.pretraining.get("augmentation", "byol") if stage == "pretrain" else "byol"
        datamodule = ColorDataModule(
            dataset_name=cfg.dataset.name,
            data_dir=cfg.dataset.data_dir,
            batch_size=cfg.experiment.batch_size,
            resize=cfg.dataset.resize,
            subset=cfg.dataset.subset,
            num_workers=cfg.experiment.num_workers,
            selected_classes=cfg.dataset.get("selected_classes", None),
            precision=cfg.architecture.precision,
            augmentation=augmentation,
            n_local_crops=cfg.pretraining.get("n_local_crops", 6),
            local_crop_size=cfg.pretraining.get("local_crop_size", 96),
            dino_global_scale=tuple(cfg.pretraining.get("dino_global_scale", [0.4, 1.0])),
            dino_local_scale=tuple(cfg.pretraining.get("dino_local_scale", [0.05, 0.4])),
            byol_scale=tuple(cfg.pretraining.get("byol_scale", [0.2, 1.0])),
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True
    else:
        raise ValueError("Invalid dataset configuration")
    datamodule.setup()
    return datamodule, input_dim, is_image


def get_pretrain_model(cfg: omegaconf.DictConfig, input_dim: int) -> Union[HyperbolicAE, HyperbolicResNet]:
    model_path = f"data/models/{get_model_path(cfg)}.pt"
    if os.path.exists(model_path) and not cfg.pretraining.overwrite:
        try:
            saved = torch.load(model_path, weights_only=False)
            if not isinstance(saved, dict):
                print(f"Loaded model from {model_path}")
                return saved
            # If it's a state dict, fall through and build a fresh model
            # (train.py will load the state dict)
        except RuntimeError:
            print(f"Warning: corrupted model file {model_path}, removing and retraining")
            os.remove(model_path)
    if cfg.architecture.model_type == "feedforward":
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
    elif cfg.pretraining.get("method", "contrastive") == "dino" and \
         cfg.architecture.model_type in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        dino_cfg = cfg.pretraining.get("dino", {})
        model = DINOHyperbolicResNet(
            # DINO-specific parameters
            dino_out_dim=dino_cfg.get("out_dim", 65536),
            dino_hidden_dim=dino_cfg.get("hidden_dim", 2048),
            dino_bottleneck_dim=dino_cfg.get("bottleneck_dim", 256),
            teacher_momentum_start=dino_cfg.get("teacher_momentum_start", 0.996),
            teacher_momentum_end=dino_cfg.get("teacher_momentum_end", 1.0),
            teacher_temperature=dino_cfg.get("teacher_temperature", 0.04),
            student_temperature=dino_cfg.get("student_temperature", 0.1),
            center_momentum=dino_cfg.get("center_momentum", 0.9),
            n_local_crops=cfg.pretraining.get("n_local_crops", 6),
            warmup_teacher_temp=dino_cfg.get("warmup_teacher_temp", 0.04),
            warmup_teacher_temp_epochs=dino_cfg.get("warmup_teacher_temp_epochs", 30),
            max_epochs=cfg.pretraining.epochs,
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
            # Layer params
            clamping_factor=cfg.architecture.hyperbolic_clamping,
            smoothing_factor=cfg.architecture.hyperbolic_smoothing,
            hyperboloid_layer_learnable_scale=cfg.architecture.hyperboloid_layer_learnable_scale,
            FHCNN_normalize=cfg.architecture.FHCNN_normalize,
            # ResNet params
            projection_layer_dims=cfg.architecture.projection_layer_dims,
            input_size=cfg.dataset.resize,
            contrastive_loss_enabled=cfg.pretraining.contrastive_loss.enabled,
            contrastive_weight=cfg.pretraining.contrastive_loss.weight,
            contrastive_temperature=cfg.pretraining.contrastive_loss.temperature,
        )
    elif cfg.architecture.model_type in [
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
        "dino_vitb16", "dino_vitb8", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14",
    ]:
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
            supervised=cfg.pretraining.supervised,
            pretrained_weights_path=cfg.pretraining.get("pretrained_weights_path", None),
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
        hyperbolic_loss_start_temperature=cfg.clustering.hyperbolic_loss.start_temperature,
        hyperbolic_loss_end_temperature=cfg.clustering.hyperbolic_loss.end_temperature,
        hyperbolic_loss_temperature_interval=cfg.clustering.hyperbolic_loss.temperature_interval,
        lca_version=cfg.clustering.hyperbolic_loss.lca_version,
        loss_version=cfg.clustering.hyperbolic_loss.loss_version,
        sampling_strategy=cfg.clustering.hyperbolic_loss.sampling_strategy,
        sampling_eps=cfg.clustering.hyperbolic_loss.sampling_eps,
        sampling_delta=cfg.clustering.hyperbolic_loss.sampling_delta,
        sampling_alpha=cfg.clustering.hyperbolic_loss.sampling_alpha,
        uniform_sample_size=cfg.clustering.hyperbolic_loss.uniform_sample_size,
        log_approximation_diff=cfg.clustering.hyperbolic_loss.log_approximation_diff,
        # Contrastive Loss parameters
        contrastive_loss_enabled=cfg.clustering.contrastive_loss.enabled,
        contrastive_temperature=cfg.clustering.contrastive_loss.temperature,
        contrastive_weight=cfg.clustering.contrastive_loss.weight,
        contrastive_euclidean=cfg.clustering.contrastive_loss.get("euclidean", False),
        # Reconstruction Loss parameters
        recon_weight=cfg.clustering.reconstruction_loss.weight,
        # Regularizer parameters
        regularize_to_origin_enabled=cfg.clustering.regularizer.enabled,
        regularizer_weight=cfg.clustering.regularizer.weight,
        regularizer_type=cfg.clustering.regularizer.get("type", "origin"),
        head_warmup_epochs=cfg.clustering.get("head_warmup_epochs", 0),
        # Backbone kNN SNN similarity parameters
        use_backbone_knn_similarities=cfg.clustering.hyperbolic_loss.get("use_backbone_knn_similarities", False),
        snn_k=cfg.clustering.hyperbolic_loss.get("snn_k", 10),
        freeze_encoder=cfg.clustering.get("freeze_encoder", False),
        unfreeze_backbone_from=cfg.clustering.get("unfreeze_backbone_from", None),
        backbone_lr_factor=cfg.clustering.get("backbone_lr_factor", 0.1),
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
