import os
from hashlib import md5

import omegaconf
import torch
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from src.models.resnet34 import HyperbolicResNet34
from src.models.feedforward import HyperbolicAE
from src.models.resnet18 import HyperbolicResNet18
from src.datasets.datamodule import ColorDataModule, GrayScaleDataModule, TabularDataModule



def get_model_path(cfg):
    attr = {
        "seed" : cfg.seed,
        "pretrain_epochs" : cfg.pretrain_epochs,
        "clustering_epochs" : cfg.clustering_epochs,
        "embedding_dim" : cfg.embedding_dim,
        "ae_type" : cfg.ae_type,
        "hidden_dims" : cfg.hidden_dims,
        "learning_rate" : cfg.learning_rate,
    }
    return md5(str(attr).encode()).hexdigest()

def get_datamodule(cfg):
    # Determine dataset type and create appropriate datamodule
    if hasattr(cfg.dataset, 'type') and cfg.dataset.type == 'tabular':
        datamodule = TabularDataModule(
            dataset_name=cfg.dataset.name,
            batch_size=cfg.dataset.batch_size,
            num_workers=cfg.num_workers,
            train_ratio=cfg.dataset.train_ratio,
            aug_noise_std=cfg.dataset.aug_noise_std
        )
        input_dim = cfg.dataset.input_dim
        is_image = False
    elif hasattr(cfg.dataset, 'channels') and cfg.dataset.channels == 1:
        datamodule = GrayScaleDataModule(
            dataset_name=cfg.dataset.name,
            batch_size=cfg.batch_size,
            data_dir=cfg.dataset.data_dir,
            resize=cfg.dataset.resize,
            num_workers=cfg.num_workers,
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True
    elif hasattr(cfg.dataset, 'channels') and cfg.dataset.channels == 3:
        datamodule = ColorDataModule(
            dataset_name=cfg.dataset.name,
            batch_size=cfg.batch_size,
            data_dir=cfg.dataset.data_dir,
            resize=cfg.dataset.resize,
            num_workers=cfg.num_workers,
            subset=cfg.dataset.subset,
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True
    else:
        raise ValueError("Invalid dataset configuration")

    datamodule.setup()
    return datamodule, input_dim, is_image

def get_model(cfg, input_dim, is_image, manifold):
    if os.path.exists(f"data/models/{get_model_path(cfg)}.pt") and not cfg.overwrite:
        model = torch.load(f"data/models/{get_model_path(cfg)}.pt", weights_only=False)
        print("Loaded model from {}".format(f"data/models/{get_model_path(cfg)}.pt"))
    elif cfg.ae_type == 'feedforward':
        model = HyperbolicAE(
            input_dim=input_dim,
            hidden_dims=cfg.hidden_dims,
            latent_dim=cfg.embedding_dim,
            learning_rate=cfg.learning_rate,
            hyphc_weight=cfg.triplet_loss_weight,
            magnitude_curvature=cfg.magnitude_curvature,
            is_image=is_image,
            log_interval=cfg.log_interval,
            pretrain_epochs=cfg.pretrain_epochs,
            wandb_enabled=cfg.wandb.enabled,
            plot_label_type=cfg.wandb.plot_label_type,
        )
    elif cfg.ae_type == 'convolutional':
        raise NotImplementedError("Convolutional autoencoder not implemented yet")
    elif cfg.ae_type == 'variational':
        raise NotImplementedError("Variational autoencoder not implemented yet")
    elif cfg.ae_type == 'resnet18' or cfg.ae_type == 'resnet':
        model = HyperbolicResNet18(
            input_dim=input_dim,
            latent_dim=cfg.embedding_dim,
            learning_rate=cfg.learning_rate,
            hyphc_weight=cfg.triplet_loss_weight,
            magnitude_curvature=cfg.magnitude_curvature,
            is_image=is_image,
            log_interval=cfg.log_interval,
            pretrain_epochs=cfg.pretrain_epochs,
            wandb_enabled=cfg.wandb.enabled,
            plot_label_type=cfg.wandb.plot_label_type,
            triplet_sampled_n=cfg.triplet_sampled_n,
            loss_batch_size=cfg.loss_batch_size,
        )
    elif cfg.ae_type == 'resnet34':
        model = HyperbolicResNet34(
            manifold=manifold,
            latent_dim=cfg.embedding_dim,
            learning_rate=cfg.learning_rate,
            log_interval=cfg.log_interval,
        )
            
    else:
        raise ValueError("Invalid autoencoder type")
    
    return model
    
def get_logger(cfg):
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            save_dir=cfg.wandb.save_dir,
            dir=cfg.wandb.save_dir,
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            tags=cfg.wandb.tags,
        )

    else:
        logger = CSVLogger(save_dir=cfg.wandb.save_dir, name=cfg.wandb.project)

    return logger