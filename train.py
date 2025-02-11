import warnings
import omegaconf
import torch
import wandb
from src.hyperbolic_math.src.manifolds.poincare import PoincareBall
from src.models.resnet34 import HyperbolicResNet34
from src.models.feedforward import HyperbolicAE
from src.models.resnet18 import HyperbolicResNet18
from src.models.shadowCones import HyperbolicContrastiveModel

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.datasets.datamodule import ColorDataModule, GrayScaleDataModule, TabularDataModule
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from lightning.pytorch import seed_everything
from threadpoolctl import threadpool_limits



@hydra.main(version_base=None, config_path="configs", config_name="config")
@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def main(cfg: DictConfig) -> None:
    print(cfg, flush=True)
    if cfg.seed >= 0:
        seed_everything(cfg.seed)   
    else:
        print("Warning: Running without setting seed!")
    torch.set_float32_matmul_precision('medium')

    warnings.filterwarnings("ignore", ".*is smaller than the logging interval Trainer.*")
    
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

    
    trainer = pl.Trainer(logger=logger,
                        max_epochs=cfg.pretrain_epochs,
                        devices='auto',
                        accelerator='auto',
                        deterministic=True,
                        enable_checkpointing=False,
                        )
   

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    manifold = PoincareBall(c=torch.Tensor([cfg.magnitude_curvature,]).to(device))

    if cfg.ae_type == 'feedforward':
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

    trainer.fit(model, datamodule)
    cone_model = HyperbolicContrastiveModel(
        resnet=model,
        manifold=manifold,
        light_radius=cfg.light_radius,
        temperature=cfg.temperature,
        contrastive_weight=cfg.contrastive_weight,
    )
    trainer = pl.Trainer(logger=logger,
                        max_epochs=cfg.clustering_epochs,
                        devices='auto',
                        accelerator='auto',
                        deterministic=True,
                        enable_checkpointing=False,
                        )
    trainer.fit(cone_model, datamodule)
    wandb.finish()


if __name__ == '__main__':
    main()
