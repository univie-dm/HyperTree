from src.models.utils import get_dtype_from_precision
from src.utils.utils import (
    get_datamodule,
    get_pretrain_model,
    get_clustering_model,
    get_model_path,
    get_logger,
)
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from threadpoolctl import threadpool_limits
import hydra
import os
import traceback
import torch
import wandb
import warnings
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from icecream import install  # type: ignore

install()


@hydra.main(version_base=None, config_path="configs", config_name="config")
@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def main(cfg: DictConfig) -> None:
    try:
        print(cfg)
        if cfg.experiment.seed >= 0:
            seed_everything(cfg.experiment.seed)
        else:
            print("Warning: Running without setting seed!")
        if cfg.architecture.precision == "float32":
            torch.set_float32_matmul_precision("medium")
            torch.set_default_dtype(torch.float32)
        elif cfg.architecture.precision == "float64":
            torch.set_float32_matmul_precision("high")
            torch.set_default_dtype(torch.float64)
        else:
            raise ValueError(f"Precision {cfg.architecture.precision} not supported")

        if (cfg.pretraining.load_supervised_weights or cfg.pretraining.supervised) and \
           cfg.architecture.model_type not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            raise ValueError("Supervised and supervised weights can only be loaded for ResNet models")

        warnings.filterwarnings(
            "ignore", ".*is smaller than the logging interval Trainer.*"
        )
        torch.set_default_dtype(get_dtype_from_precision(cfg.architecture.precision))
        logger = get_logger(cfg)

        pretrain_ckpt_dir = f"./data/checkpoints/pretrain_{get_model_path(cfg)}"
        if cfg.dataset.name == 'imagenet':
            os.makedirs(pretrain_ckpt_dir, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=pretrain_ckpt_dir,
                save_last=True,
                every_n_epochs=10,
            )

        trainer = pl.Trainer(
            logger=logger,
            max_epochs=cfg.pretraining.epochs,
            devices="auto",
            accelerator="auto",
            deterministic=True if cfg.experiment.seed >= 0 else False,
            enable_checkpointing=True if cfg.dataset.name == 'imagenet' else False,
            default_root_dir=None,
            gradient_clip_algorithm="value"
            if cfg.experiment.gradient_clip_value is not None
            else None,
            gradient_clip_val=cfg.experiment.gradient_clip_value
            if cfg.experiment.gradient_clip_value is not None
            else 1.0,
            log_every_n_steps=None,
            callbacks=[checkpoint_callback] if cfg.dataset.name == 'imagenet' else None,
        )
        datamodule, input_dim, _ = get_datamodule(cfg, stage="pretrain")
        pretrain_model = get_pretrain_model(cfg, input_dim)

        if (not os.path.exists(f"data/models/{get_model_path(cfg)}.pt")
            or cfg.pretraining.overwrite):
            print("Starting pretraining", flush=True)
            ckpt_path = None
            if cfg.dataset.name == 'imagenet':
                last_ckpt = os.path.join(pretrain_ckpt_dir, "last.ckpt")
                if os.path.exists(last_ckpt):
                    print(f"Resuming pretraining from checkpoint: {last_ckpt}", flush=True)
                    ckpt_path = last_ckpt
            trainer.fit(pretrain_model, datamodule, ckpt_path=ckpt_path)
            torch.save(pretrain_model, f"data/models/{get_model_path(cfg)}.pt")
        else:
            pretrain_model = torch.load(f"data/models/{get_model_path(cfg)}.pt", weights_only=False)

        print("Finished pretraining", flush=True)
        if cfg.experiment.seed >= 0:
            seed_everything(cfg.experiment.seed)
        print("Starting clustering", flush=True)
        datamodule, input_dim, _ = get_datamodule(cfg, stage="clustering")

        accumulate_grad_batches = max(
            1,
            min(cfg.clustering.accumulate_gradients, len(datamodule.train_dataloader())))

        # Determine checkpoint directory for clustering
        clustering_checkpoint_dir = None
        if cfg.clustering.get("enable_checkpointing", False):
            clustering_checkpoint_dir = f"./data/checkpoints/clustering_{cfg.dataset.name}_{get_model_path(cfg)}"
            os.makedirs(clustering_checkpoint_dir, exist_ok=True)
            print(f"Checkpointing enabled for clustering. Checkpoints will be saved to: {clustering_checkpoint_dir}",
                  flush=True)

        clustering_model = get_clustering_model(cfg, pretrain_model)
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=cfg.clustering.epochs,
            devices="auto",
            accelerator="auto",
            deterministic=True if cfg.experiment.seed >= 0 else False,
            enable_checkpointing=False,
            default_root_dir=None,
            accumulate_grad_batches=(accumulate_grad_batches),
            gradient_clip_algorithm="norm"
            if cfg.experiment.gradient_clip_value is not None
            else None,
            gradient_clip_val=cfg.experiment.gradient_clip_value
            if cfg.experiment.gradient_clip_value is not None
            else 1.0,
        )
        trainer.fit(clustering_model, datamodule)
        wandb.finish()
    except Exception as e:
        wandb.finish(exit_code=-1)
        print(f"Error occurred: {e}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
