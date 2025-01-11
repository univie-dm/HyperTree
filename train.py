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
   

    datamodule = GrayScaleDataModule(
        dataset_name='mnist',
        batch_size=512,
        data_dir='data',
        num_workers=16,
    )
    datamodule.setup()

    '''
    datamodule = TabularDataModule(
        dataset_name='iris',
        batch_size=32,
        num_workers=16,
    )
    '''

    model = HyperbolicAE(
        input_dim=28*28,
        hidden_dims=[512, 256, 64],
        latent_dim=10
    )

    trainer = pl.Trainer(logger=logger,
                         max_epochs=10,
                         devices='auto',
                         accelerator='auto',
                         deterministic=True,
                         enable_checkpointing=False,
                         )

    trainer.fit(model, datamodule)
    wandb.finish()


if __name__ == '__main__':
    main()
