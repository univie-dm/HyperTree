# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- **Always use `uv run`** — never bare `python`. This is an HPC login node; use `uv run python ...` or `uv run pytest ...`.
- Python >=3.11, dependencies managed via `uv` with `uv.lock`.
- Install/sync: `uv sync`

## Common Commands

```bash
# Training (Hydra CLI)
uv run python train.py --config-name config
uv run python train.py --config-name config dataset=cifar100 architecture.model_type=resnet50
uv run python train.py --config-name imagenet_dino

# Tests (hyperbolic math library only)
uv run pytest src/hyperbolic_math/tests/

# SLURM submission
sbatch sbatch.sh
```

## Architecture

**Two-stage pipeline** orchestrated by `train.py` (Hydra `@hydra.main`):

1. **Pretraining** — Encoder (ResNet or feedforward) + projection head trained with contrastive or DINO self-distillation loss. Pretrained weights saved to `data/models/{hash}.pt`.
2. **Clustering** — `HyperbolicHCModel` wraps the pretrained encoder, attaches a hyperbolic clustering head, and trains with hyperbolic clustering loss (+ optional contrastive/reconstruction losses). Evaluates dendrogram purity, k-means, spectral clustering, nearest neighbors.

**Model hierarchy** (all PyTorch Lightning modules):
- `AbstractModel` (`src/models/abstract.py`) — base class defining `encode()`, `decode()`, `forward()`, optimizer setup, evaluation
- `HyperbolicAE` (`feedforward.py`) — autoencoder for tabular/small data
- `HyperbolicResNet` (`resnet.py`) — ResNet backbone + hyperbolic projection
- `DINOHyperbolicResNet` (`dino.py`) — DINO teacher-student variant with multi-crop
- `HyperbolicHCModel` (`hyphc.py`) — clustering wrapper used in stage 2

**Manifolds** (via `src/hyperbolic_math/` submodule): Euclidean, Hyperboloid, PoincareBall. Riemannian optimizers in `hyperbolic_math/optim/`.

**Config system**: Hydra configs in `configs/`. Base config is `configs/config.yaml`. Dataset-specific overrides in `configs/dataset/`. All parameters overridable via CLI.

**Data**: `ColorDataModule` handles image datasets (CIFAR, ImageNet) with augmentation strategies (DINO multi-crop, BYOL, SimCLR). `TabularDataModule` for small datasets.

**Factories**: `get_model()` and `get_datamodule()` in `src/utils/utils.py` dispatch based on config.

**Evaluation**: `src/evaluation.py` — dendrogram purity, clustering metrics, kNN accuracy, 2D Poincaré ball visualization. Called at end of training and optionally during validation.
