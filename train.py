import os
import warnings
import omegaconf
import torch
import wandb
from src.hyperbolic_math.src.manifolds.poincare import PoincareBall
from src.models.shadowCones import HyperbolicContrastiveModel

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

from lightning.pytorch import seed_everything
from threadpoolctl import threadpool_limits
from src.utils import get_datamodule, get_model, get_model_path, get_logger



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
    manifold = PoincareBall(c=torch.Tensor([cfg.magnitude_curvature,]).to(device), trainable_c=False)
    
    model = get_model(cfg, input_dim, is_image, manifold)

    if not os.path.exists(f"data/models/{get_model_path(cfg)}.pt") or cfg.overwrite:
        print("Starting pretraining")
        trainer.fit(model, datamodule)
        torch.save(model, f"data/models/{get_model_path(cfg)}.pt")
        print("Saved model to {}".format(f"data/models/{get_model_path(cfg)}.pt"))

    print("Finished pretraining")
    print("Starting clustering")

    manifold = PoincareBall(c=torch.Tensor([cfg.magnitude_curvature,]).to(device), trainable_c=cfg.trainable_curvature)

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
