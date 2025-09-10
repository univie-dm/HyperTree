import hydra
import os
import torch
import wandb
import warnings
import pytorch_lightning as pl

from icecream import install
install()

from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits
from src.utils import get_datamodule, get_pretrain_model, get_clustering_model, get_model_path, get_logger


@hydra.main(version_base=None, config_path="configs", config_name="config")
@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def main(cfg: DictConfig) -> None:
    print(cfg, flush=True)
    if cfg.experiment.seed >= 0:
        seed_everything(cfg.experiment.seed)
    else:
        print("Warning: Running without setting seed!", flush=True)
    if cfg.architecture.precision == 'float32':
        torch.set_float32_matmul_precision('medium')
        torch.set_default_dtype(torch.float32)
    elif cfg.architecture.precision == 'float64':
        torch.set_float32_matmul_precision('high')
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Precision {cfg.architecture.precision} not supported")

    warnings.filterwarnings("ignore", ".*is smaller than the logging interval Trainer.*")
    logger = get_logger(cfg)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=cfg.pretraining.epochs,
                         devices='auto',
                         accelerator='auto',
                         deterministic=True,
                         enable_checkpointing=False,
                         gradient_clip_algorithm='norm' if cfg.experiment.gradient_clip_value is not None else None,
                         gradient_clip_val=cfg.experiment.gradient_clip_value if cfg.experiment.gradient_clip_value is not None else 1.0,
                         )
    datamodule, input_dim, _ = get_datamodule(cfg, stage='pretrain')
    pretrain_model = get_pretrain_model(cfg, input_dim)

    if not os.path.exists(f"data/models/{get_model_path(cfg)}.pt") or cfg.pretraining.overwrite:
        print("Starting pretraining", flush=True)
        trainer.fit(pretrain_model, datamodule)
        torch.save(pretrain_model, f"data/models/{get_model_path(cfg)}.pt")
    else:
        pretrain_model = torch.load(f"data/models/{get_model_path(cfg)}.pt", weights_only=False)

    print("Finished pretraining", flush=True)
    print("Starting clustering", flush=True)
    datamodule, input_dim, _ = get_datamodule(cfg, stage='clustering')

    accumulate_grad_batches = max(1, min(cfg.clustering.accumulate_gradients, len(datamodule.train_dataloader())))

    clustering_model = get_clustering_model(cfg, pretrain_model)
    trainer = pl.Trainer(logger=logger,
                         max_epochs=cfg.clustering.epochs,
                         devices='auto',
                         accelerator='auto',
                         deterministic=True,
                         enable_checkpointing=False,
                         accumulate_grad_batches=(accumulate_grad_batches),
                         gradient_clip_algorithm='norm' if cfg.experiment.gradient_clip_value is not None else None,
                         gradient_clip_val=cfg.experiment.gradient_clip_value if cfg.experiment.gradient_clip_value is not None else 1.0,
                         )
    trainer.fit(clustering_model, datamodule)
    wandb.finish()


if __name__ == '__main__':
    main()
