import omegaconf
import torch
import wandb
from src.brb import BRBCallback
from src.progress import RichProgressBar
from src.dc_algorithms.dcn import DCN
from src.dc_algorithms.dec import DEC
from src.dc_algorithms.idec import IDEC

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from src.models.feedforward import Autoencoder
from src.datasets.datamodule import ColorDataModule, GrayScaleDataModule
from src.eval import MetricsCallback, calculate_metrics
from pytorch_lightning.loggers import WandbLogger, CSVLogger

from src.utils import set_seed, load_pretrained_model, save_pretrained_model
from lightning.pytorch import seed_everything
from threadpoolctl import threadpool_limits


@hydra.main(version_base=None, config_path="configs", config_name="config")
@threadpool_limits.wrap(limits=45, user_api="openmp")
@threadpool_limits.wrap(limits=1, user_api="blas")
def main(cfg: DictConfig) -> None:
    print(cfg, flush=True)
    cluster_init_seed = None
    if cfg.seed >= 0:
        set_seed(cfg.seed)
        seed_everything(cfg.seed)
        cluster_init_seed = cfg.seed        
    else:
        print("Warning: Running without setting seed!")
    torch.set_float32_matmul_precision('medium')
    if cfg.dataset.channels == 1:
        datamodule = GrayScaleDataModule(
            cfg.dataset.name,
            batch_size=cfg.batch_size,
            resize=cfg.dataset.resize,
            subset=cfg.dataset.subset,
            data_dir=cfg.dataset.data_dir,
            num_workers=cfg.num_workers,
        )
    elif cfg.dataset.channels == 3:
        datamodule = ColorDataModule(
            cfg.dataset.name,
            batch_size=cfg.batch_size,
            resize=cfg.dataset.resize,
            subset=cfg.dataset.subset,
            data_dir=cfg.dataset.data_dir,
            num_workers=cfg.num_workers,
        )
    else:
        raise ValueError(f"Unknown number of channels {cfg.dataset.channels}")
    
    datamodule.setup()
    
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
    # First stage: Pretrain autoencoder or load if exists
    ae = Autoencoder(
        input_dim=cfg.dataset.resize * cfg.dataset.resize * cfg.dataset.channels,
        learning_rate=cfg.pretrain_lr,
        embedding_dim=cfg.embedding_dim,
    )
    
    loaded, ae = load_pretrained_model(ae, cfg)
    if not loaded:
        progress_bar = RichProgressBar(total_epochs=cfg.pretrain_epochs, mode='pretrain')
        trainer = pl.Trainer(
            max_epochs=cfg.pretrain_epochs,
            logger=logger,
            accelerator='auto',
            enable_checkpointing=False,
            callbacks=[progress_bar],
            gradient_clip_val=500,
            deterministic=True,)
        trainer.fit(ae, datamodule)
        save_pretrained_model(ae, cfg)

    # Second stage: Clustering
    if cfg.dc_algorithm == 'dec':
        dc = DEC(
            ae,
            n_clusters=cfg.n_clusters,
            learning_rate=cfg.clustering_lr,
            cluster_loss_weight=cfg.cluster_loss_weight,
            augmentation_invariance=cfg.augmentation_invariance,
            seed=cluster_init_seed,
            log_interval=cfg.log_interval,
        )
    elif cfg.dc_algorithm == 'idec':
        dc = IDEC(
            ae,
            cluster_loss_weight=cfg.cluster_loss_weight,
            n_clusters=cfg.n_clusters,
            learning_rate=cfg.clustering_lr,
            augmentation_invariance=cfg.augmentation_invariance,
            seed=cluster_init_seed,
            log_interval=cfg.log_interval
        )
    elif cfg.dc_algorithm == 'dcn':
        dc = DCN(
            ae,
            cluster_loss_weight=cfg.cluster_loss_weight,
            n_clusters=cfg.n_clusters,
            learning_rate=cfg.clustering_lr,
            augmentation_invariance=cfg.augmentation_invariance,
            seed=cluster_init_seed,
            log_interval=cfg.log_interval
        )
    else:
        raise ValueError(f"Unknown algorithm {cfg.dc_algorithm}")
    
    if cfg.clustering_epochs <= 0:
        print("Skipping clustering stage")
        return

    dc.initialize_clusters(datamodule.train_dataloader())

    
    callbacks = []
    
    # Only add BRB callback if enabled and properly configured
    if (cfg.brb.enabled and 
        cfg.brb.reset_interval > 0 and 
        any([cfg.brb.reset_weights, cfg.brb.reset_momentum, 
             cfg.brb.reset_momentum_counts, cfg.brb.recluster,
             cfg.brb.recalculate_centers, cfg.brb.reset_batchnorm])):
        callbacks.append(BRBCallback(
            reset_interval=cfg.brb.reset_interval,
            reset_weights=cfg.brb.reset_weights,
            reset_momentum=cfg.brb.reset_momentum,
            reset_momentum_counts=cfg.brb.reset_momentum_counts,
            recluster=cfg.brb.recluster,
            recalculate_centers=cfg.brb.recalculate_centers,
            reset_interpolation_factor=cfg.brb.reset_interpolation_factor,
            reset_batchnorm=cfg.brb.reset_batchnorm,
            n_clusters=cfg.n_clusters,
            clustering_method=cfg.brb.clustering_method,
            subsample_size=cfg.brb.subsample_size
        ))

    progress_bar = RichProgressBar(total_epochs=cfg.clustering_epochs, mode='clustering')
    callbacks.append(progress_bar)
    trainer = pl.Trainer(
        max_epochs=cfg.clustering_epochs,
        logger=logger, 
        accelerator='auto',
        callbacks=callbacks,
        enable_checkpointing=False,
        gradient_clip_val=500,
        deterministic=True,
    )
    trainer.fit(dc, datamodule)
    
    wandb.finish()


if __name__ == '__main__':
    main()
