from src.models.utils import get_dtype_from_precision
from src.utils.utils import (
    get_datamodule,
    get_pretrain_model,
    get_clustering_model,
    get_model_path,
    get_run_path,
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

        # cuDNN benchmark mode: lets cuDNN auto-tune the fastest conv algorithm
        # for each unique input shape. Off by default in PyTorch. (Conv TF32 is
        # already on by default in torch 2.9 via cudnn.conv.fp32_precision='tf32';
        # matmul TF32 is set above. Don't mix the legacy allow_tf32 API with the
        # new one — torch 2.9 errors out.)
        if cfg.experiment.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True

        warnings.filterwarnings(
            "ignore", ".*is smaller than the logging interval Trainer.*"
        )
        torch.set_default_dtype(get_dtype_from_precision(cfg.architecture.precision))

        # Clustering checkpoints are keyed by the whole config, so that starting the same configuration again
        # (Slurm requeue after preemption, resubmission after the time limit) resumes from the last checkpoint
        # and continues its W&B run
        enable_clustering_ckpt = cfg.clustering.enable_checkpointing
        clustering_ckpt_dir = f"./data/checkpoints/clustering_{get_run_path(cfg)}"
        clustering_last_ckpt = os.path.join(clustering_ckpt_dir, "last.ckpt")
        resume_clustering = enable_clustering_ckpt and os.path.exists(clustering_last_ckpt)
        wandb_id = None
        if enable_clustering_ckpt and cfg.wandb.enabled:
            wandb_id_file = os.path.join(clustering_ckpt_dir, "wandb_id")
            if resume_clustering and os.path.exists(wandb_id_file):
                with open(wandb_id_file) as f:
                    wandb_id = f.read().strip()
            else:
                wandb_id = wandb.util.generate_id()
                if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                    os.makedirs(clustering_ckpt_dir, exist_ok=True)
                    with open(wandb_id_file, "w") as f:
                        f.write(wandb_id)
        logger = get_logger(cfg, wandb_id)

        model_path = f"data/models/{get_model_path(cfg)}.pt"
        pretrain_ckpt_dir = f"./data/checkpoints/pretrain_{get_model_path(cfg)}"
        enable_pretrain_ckpt = cfg.pretraining.enable_checkpointing
        checkpoint_callback = None
        if enable_pretrain_ckpt:
            os.makedirs(pretrain_ckpt_dir, exist_ok=True)
            checkpoint_callback = ModelCheckpoint(
                dirpath=pretrain_ckpt_dir,
                save_last=True,
                every_n_epochs=10,
            )

        pretrain_accum = max(1, cfg.pretraining.accumulate_gradients)
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
        # Default = "32-true" (Lightning's own default). bf16-mixed cuts memory
        # and gives a big tensor-core speedup on the conv-heavy backbone.
        pretrain_precision = cfg.experiment.training_precision
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
        datamodule = get_datamodule(cfg, stage="pretrain")
        pretrain_model = get_pretrain_model(cfg)

        if not os.path.exists(model_path) or cfg.pretraining.overwrite:
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
                    print("LR scheduler mismatch in checkpoint, loading weights only and restarting training", flush=True)
                    ckpt = torch.load(ckpt_path, weights_only=False)
                    pretrain_model.load_state_dict(ckpt["state_dict"])
                    trainer.fit(pretrain_model, datamodule)
                else:
                    raise
            torch.save(pretrain_model.state_dict(), model_path)
        else:
            if is_rank0: print(f"Loading pretrained model from {model_path}", flush=True)
            pretrain_model.load_state_dict(torch.load(model_path, weights_only=False))

        print("Finished pretraining", flush=True)
        if cfg.experiment.seed >= 0:
            seed_everything(cfg.experiment.seed)
        print("Starting clustering", flush=True)
        datamodule = get_datamodule(cfg, stage="clustering")

        accumulate_grad_batches = max(
            1,
            min(cfg.clustering.accumulate_gradients, len(datamodule.train_dataloader())))

        clustering_model = get_clustering_model(cfg, pretrain_model)

        clustering_callbacks = []
        if enable_clustering_ckpt:
            clustering_callbacks.append(ModelCheckpoint(
                dirpath=clustering_ckpt_dir,
                save_last="link",
                every_n_epochs=cfg.clustering.checkpoint_interval,
            ))
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=cfg.clustering.epochs,
            accelerator="gpu" if gpu_count > 0 else "cpu",
            devices=n_devices,
            strategy="ddp_find_unused_parameters_true" if use_ddp else "auto",
            accumulate_grad_batches=accumulate_grad_batches,
            deterministic=cfg.experiment.get("deterministic", False),
            enable_checkpointing=enable_clustering_ckpt,
            callbacks=clustering_callbacks,
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
        if resume_clustering:
            print(f"Resuming clustering from checkpoint: {clustering_last_ckpt}", flush=True)
        # Our own checkpoint, which also holds the (OmegaConf) hyperparameters, so not a weights-only load
        trainer.fit(clustering_model, datamodule, ckpt_path=clustering_last_ckpt if resume_clustering else None,
                    weights_only=False)
        wandb.finish()
    except Exception as e:
        wandb.finish(exit_code=-1)
        print(f"Error occurred: {e}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
