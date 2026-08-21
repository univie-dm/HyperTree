"""Measure peak GPU memory of both training stages for a given config.

Mirrors train.py's model/datamodule construction and Trainer settings, but
runs only a handful of batches per stage. Reports peak *training-step*
memory: the counter is reset after AbstractModel.on_train_start's evaluation
pass (abstract.py:634) so the number reflects the training step itself,
which is what actually OOMs, not the evaluation.

Run under srun/sbatch, never on the login node. Example:

    uv run python scripts/measure_memory.py --config-name config \
        dataset=cifar100 experiment.batch_size=64 wandb.enabled=False
"""
import json
import os
import sys
from pathlib import Path

# Run as a script from anywhere: put the repo root on sys.path so `src` imports
# resolve the same way they do for train.py at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import pytorch_lightning as pl
import torch
from lightning.pytorch import seed_everything
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback

from src.models.utils import get_dtype_from_precision
from src.utils.utils import (
    get_clustering_model,
    get_datamodule,
    get_logger,
    get_pretrain_model,
)

# Batches to run per stage. Peak memory is reached on the first step, so a
# handful is plenty; more only costs wall time.
N_BATCHES = 5


class PeakMemory(Callback):
    """Record peak allocated/reserved memory across training steps only."""

    def __init__(self) -> None:
        self.allocated = 0
        self.reserved = 0
        self._reset_done = False

    def on_train_batch_start(self, *args, **kwargs) -> None:
        # Reset once, after on_train_start's evaluation pass, so the peak we
        # report is the training step and not the eval forward passes.
        if not self._reset_done:
            torch.cuda.reset_peak_memory_stats()
            self._reset_done = True

    def on_train_batch_end(self, *args, **kwargs) -> None:
        self.allocated = max(self.allocated, torch.cuda.max_memory_allocated())
        self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())


def _trainer(cfg: DictConfig, precision: str, callback: Callback) -> pl.Trainer:
    return pl.Trainer(
        logger=get_logger(cfg),
        max_epochs=1,
        limit_train_batches=N_BATCHES,
        num_sanity_val_steps=0,
        accelerator="gpu",
        devices=1,
        precision=precision,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accumulate_grad_batches=max(1, cfg.pretraining.get("accumulate_gradients", 1)),
        gradient_clip_algorithm="value"
        if cfg.experiment.gradient_clip_value is not None
        else None,
        gradient_clip_val=cfg.experiment.gradient_clip_value
        if cfg.experiment.gradient_clip_value is not None
        else 1.0,
        callbacks=[callback],
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device visible — submit this via srun/sbatch.")

    if cfg.experiment.seed >= 0:
        seed_everything(cfg.experiment.seed)
    # Mirror train.py:75-100 so dtype/TF32 state matches a real run.
    if cfg.architecture.precision == "float32":
        torch.set_float32_matmul_precision("medium")
    else:
        torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(get_dtype_from_precision(cfg.architecture.precision))
    torch.backends.cudnn.benchmark = cfg.experiment.get("cudnn_benchmark", True)

    label = os.environ.get("MEASURE_LABEL", "unlabelled")
    result = {
        "label": label,
        "gpu": torch.cuda.get_device_name(0),
        "model_type": cfg.architecture.model_type,
        "batch_size": cfg.experiment.batch_size,
        "latent_dim": cfg.architecture.latent_dim,
        "training_precision": cfg.experiment.get("training_precision", "32-true"),
        "manifold_precision": cfg.architecture.manifold_precision,
    }

    datamodule, input_dim, _ = get_datamodule(cfg, stage="pretrain")
    pretrain_model = get_pretrain_model(cfg, input_dim)

    pretrain_peak = PeakMemory()
    _trainer(cfg, cfg.experiment.get("training_precision", "32-true"), pretrain_peak).fit(
        pretrain_model, datamodule
    )
    result["pretrain_gib"] = round(pretrain_peak.allocated / 2**30, 2)
    result["pretrain_reserved_gib"] = round(pretrain_peak.reserved / 2**30, 2)

    # Clustering runs at 32-true in train.py:233 — training_precision is
    # deliberately not applied to this stage.
    datamodule, _, _ = get_datamodule(cfg, stage="clustering")
    clustering_model = get_clustering_model(cfg, pretrain_model)

    clustering_peak = PeakMemory()
    _trainer(cfg, "32-true", clustering_peak).fit(clustering_model, datamodule)
    result["clustering_gib"] = round(clustering_peak.allocated / 2**30, 2)
    result["clustering_reserved_gib"] = round(clustering_peak.reserved / 2**30, 2)

    result["peak_gib"] = max(result["pretrain_gib"], result["clustering_gib"])
    result["peak_reserved_gib"] = max(
        result["pretrain_reserved_gib"], result["clustering_reserved_gib"]
    )

    os.makedirs("results/memory", exist_ok=True)
    with open(f"results/memory/{label}.json", "w") as f:
        json.dump(result, f, indent=2)
    print("MEASUREMENT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
