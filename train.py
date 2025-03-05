import hydra
import os
import torch
import wandb
import warnings
import pytorch_lightning as pl

from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits
from src.models.contrastive import HyperbolicContrastiveModel
from src.utils import get_datamodule, get_model, get_model_path, get_logger


@hydra.main(version_base=None, config_path="configs", config_name="config")
@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def main(cfg: DictConfig) -> None:
    print(cfg, flush=True)
    if cfg.seed >= 0:
        seed_everything(cfg.seed)
    else:
        print("Warning: Running without setting seed!", flush=True)
    torch.set_float32_matmul_precision('medium')

    warnings.filterwarnings("ignore", ".*is smaller than the logging interval Trainer.*")

    logger = get_logger(cfg)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=cfg.pretrain_epochs,
                         devices='auto',
                         accelerator='auto',
                         deterministic=True,
                         enable_checkpointing=False,
                         )
    datamodule, input_dim, is_image = get_datamodule(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_model = get_model(cfg, input_dim)

    if not os.path.exists(f"data/models/{get_model_path(cfg)}.pt") or cfg.overwrite:
        print("Starting pretraining", flush=True)
        trainer.fit(ae_model, datamodule)
        torch.save(ae_model, f"data/models/{get_model_path(cfg)}.pt")
    else:
        ae_model = torch.load(f"data/models/{get_model_path(cfg)}.pt", weights_only=False)

    print("Finished pretraining", flush=True)
    print("Starting clustering", flush=True)

    contrastive_model = HyperbolicContrastiveModel(
        ae_model=ae_model,
        clustering_trainable_curvature=cfg.clustering_trainable_curvature,
        contrastive_weight=cfg.contrastive_weight,
        contrastive_temperature=cfg.contrastive_temperature,
        light_radius=cfg.light_radius,
        start_epoch=cfg.pretrain_epochs
    )
    trainer = pl.Trainer(logger=logger,
                         max_epochs=cfg.clustering_epochs,
                         devices='auto',
                         accelerator='auto',
                         deterministic=True,
                         enable_checkpointing=False,
                         )
    trainer.fit(contrastive_model, datamodule)
    wandb.finish()


if __name__ == '__main__':
    main()
