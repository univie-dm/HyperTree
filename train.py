import omegaconf
import torch
import wandb
from src.models.feedforward import HyperbolicAE

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
    
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            save_dir=cfg.wandb.save_dir,
            dir=cfg.wandb.save_dir,
            config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    else:
        logger = CSVLogger(save_dir=cfg.wandb.save_dir, name=cfg.wandb.project)
   

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
    else:
        # Default to image dataset
        datamodule = GrayScaleDataModule(
            dataset_name=cfg.dataset.name,
            batch_size=cfg.batch_size,
            data_dir=cfg.dataset.data_dir,
            resize=cfg.dataset.resize,
            num_workers=cfg.num_workers,
        )
        input_dim = cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels
        is_image = True

    datamodule.setup()

    model = HyperbolicAE(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        latent_dim=cfg.embedding_dim,
        learning_rate=cfg.learning_rate,
        triplet_weight=cfg.triplet_weight,
        magnitude_curvature=cfg.magnitude_curvature,
        is_image=is_image,
        log_interval=cfg.log_interval,
    )

    trainer = pl.Trainer(logger=logger,
                         max_epochs=100,
                         devices='auto',
                         accelerator='auto',
                         deterministic=True,
                         enable_checkpointing=False,
                         )

    trainer.fit(model, datamodule)
    wandb.finish()


if __name__ == '__main__':
    main()
