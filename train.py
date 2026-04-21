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
from pytorch_lightning.plugins.environments import LightningEnvironment
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

        # cuDNN benchmark mode — biggest lever for conv-heavy multi-crop training.
        # Lets cuDNN auto-tune the fastest conv algorithm for each unique input
        # shape, critical for DINO's mixed 224+96 multi-crop sizes. Off by default
        # in PyTorch. (Conv TF32 is already on by default in torch 2.9 via
        # cudnn.conv.fp32_precision='tf32'; matmul TF32 is set above. Don't mix
        # the legacy allow_tf32 API with the new one — torch 2.9 errors out.)
        if cfg.experiment.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True

        if (cfg.pretraining.load_supervised_weights or cfg.pretraining.supervised) and \
           cfg.architecture.model_type not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
            raise ValueError("Supervised and supervised weights can only be loaded for ResNet models")

        warnings.filterwarnings(
            "ignore", ".*is smaller than the logging interval Trainer.*"
        )
        torch.set_default_dtype(get_dtype_from_precision(cfg.architecture.precision))
        logger = get_logger(cfg)

        pretrain_ckpt_dir = f"./data/checkpoints/pretrain_{get_model_path(cfg)}"
        enable_pretrain_ckpt = cfg.pretraining.get("enable_checkpointing", False)
        checkpoint_callback = None
        if enable_pretrain_ckpt:
            os.makedirs(pretrain_ckpt_dir, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=pretrain_ckpt_dir,
                save_last=True,
                every_n_epochs=10,
            )

        pretrain_accum = max(1, cfg.pretraining.get("accumulate_gradients", 1))
        gpu_count = torch.cuda.device_count()
        # When SLURM allocates multiple GPUs to a single task (ntasks-per-node=1),
        # Lightning's SLURM auto-detection conflicts with devices>1.
        # Remove the env vars so Lightning uses standard DDP instead of SLURM DDP.
        is_slurm = "SLURM_JOB_ID" in os.environ
        slurm_ntasks = int(os.environ.get("SLURM_NTASKS_PER_NODE", "0"))
        if slurm_ntasks == 1 and gpu_count > 1:
            for var in ("SLURM_NTASKS", "SLURM_NTASKS_PER_NODE", "SLURM_PROCID",
                        "SLURM_LOCALID", "SLURM_NODEID"):
                os.environ.pop(var, None)
        n_devices = gpu_count if gpu_count > 0 else 1
        use_ddp = n_devices > 1
        is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0


        # Optional Lightning autocast precision for the pretraining stage only.
        # Default = "32-true" (Lightning's own default). bf16-mixed gives a big
        # H100 tensor-core speedup on conv-heavy DINO multi-crop. Hyperbolic ops
        # are protected because PoincareBall._2manifold_dtype casts inputs back
        # to its declared FP32 dtype inside expmap/logmap (poincare.py:338).
        pretrain_precision = cfg.experiment.get("training_precision", "32-true")
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=cfg.pretraining.epochs,
            accelerator="gpu",
            devices=n_devices,
            strategy="ddp_find_unused_parameters_true" if use_ddp else "auto",
            precision=pretrain_precision,
            deterministic=cfg.experiment.get("deterministic", False),
            enable_checkpointing=enable_pretrain_ckpt,
            default_root_dir=None,
            accumulate_grad_batches=pretrain_accum,
            gradient_clip_algorithm="value"
            if cfg.experiment.gradient_clip_value is not None
            else None,
            gradient_clip_val=cfg.experiment.gradient_clip_value
            if cfg.experiment.gradient_clip_value is not None
            else 1.0,
            log_every_n_steps=None,
            callbacks=[checkpoint_callback] if checkpoint_callback else None,
        )
        print(
            f"[DDP] pretrain stage: world_size={trainer.world_size} "
            f"global_rank={trainer.global_rank} local_rank={trainer.local_rank} "
            f"strategy={type(trainer.strategy).__name__}",
            flush=True,
        )
        datamodule, input_dim, _ = get_datamodule(cfg, stage="pretrain")
        pretrain_model = get_pretrain_model(cfg, input_dim)

        # Cross-dataset transfer: load pretrained weights from an explicit path (skips pretraining)
        pretrained_model_path = cfg.pretraining.get("pretrained_model_path", None)
        if pretrained_model_path is not None:
            if is_rank0: print(f"Loading pretrained weights from: {pretrained_model_path}", flush=True)
            saved = torch.load(pretrained_model_path, weights_only=False)
            if isinstance(saved, dict):
                pretrain_model.load_state_dict(saved)
            else:
                pretrain_model = saved
        elif (not os.path.exists(f"data/models/{get_model_path(cfg)}.pt")
            or cfg.pretraining.overwrite):
            if is_rank0: print("Starting pretraining", flush=True)
            ckpt_path = None
            if enable_pretrain_ckpt:
                last_ckpt = os.path.join(pretrain_ckpt_dir, "last.ckpt")
                if os.path.exists(last_ckpt):
                    print(f"Resuming pretraining from checkpoint: {last_ckpt}", flush=True)
                    ckpt_path = last_ckpt
            try:
                trainer.fit(pretrain_model, datamodule, ckpt_path=ckpt_path)
            except KeyError as e:
                if "_schedulers" in str(e) and ckpt_path is not None:
                    print(f"LR scheduler mismatch in checkpoint, loading weights only and restarting training", flush=True)
                    ckpt = torch.load(ckpt_path, weights_only=False)
                    pretrain_model.load_state_dict(ckpt["state_dict"])
                    trainer.fit(pretrain_model, datamodule)
                else:
                    raise
            torch.save(pretrain_model.state_dict(), f"data/models/{get_model_path(cfg)}.pt")
        else:
            saved = torch.load(f"data/models/{get_model_path(cfg)}.pt", weights_only=False)
            if isinstance(saved, dict):
                pretrain_model.load_state_dict(saved)
            else:
                pretrain_model = saved

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
            accelerator="gpu" if gpu_count > 0 else "cpu",
            devices=n_devices,
            strategy="ddp_find_unused_parameters_true" if use_ddp else "auto",
            accumulate_grad_batches=accumulate_grad_batches,
            deterministic=cfg.experiment.get("deterministic", False),
            enable_checkpointing=False,
            default_root_dir=None,
            gradient_clip_algorithm="norm"
            if cfg.experiment.gradient_clip_value is not None
            else None,
            gradient_clip_val=cfg.experiment.gradient_clip_value
            if cfg.experiment.gradient_clip_value is not None
            else 1.0,
            plugins=[LightningEnvironment()] if is_slurm else [],
        )
        print(
            f"[DDP] clustering stage: world_size={trainer.world_size} "
            f"global_rank={trainer.global_rank} local_rank={trainer.local_rank} "
            f"strategy={type(trainer.strategy).__name__}",
            flush=True,
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
