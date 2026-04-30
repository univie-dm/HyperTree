# Hyperbolic Hierarchical Clustering

Two-stage pipeline for hyperbolic hierarchical clustering on image datasets:
self-supervised pretraining of an encoder, followed by attaching a hyperbolic
clustering head and training with hyperbolic clustering losses on a Poincaré
or Lorentz manifold.

## Installation

Requires Python >= 3.11. Dependencies are managed with
[uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs the top-level package along with the inlined `hyperbolic_math`
library at `src/hyperbolic_math/`.

## Datasets

Place datasets under `./data/` (the default path used by the configs). The
repository ships with the WordNet / iNat hierarchy files used for evaluation:

- `data/cifar100_wordnet_hierarchy.txt`
- `data/imagenet_wordnet_hierarchy.txt`
- `data/tiny_imagenet_wordnet_hierarchy.txt`
- `data/inat18_taxonomy_hierarchy.txt`

For ImageNet and iNat18, point the dataset config at the directory holding
the standard train/val splits. Tiny-ImageNet and CIFAR-100 will be
downloaded automatically by `torchvision`.

## Training

The pipeline is driven by `train.py` via [Hydra](https://hydra.cc/). The base
config is `configs/config.yaml`; dataset overrides live in `configs/dataset/`.

```bash
# CIFAR-100 with default config
uv run python train.py --config-name config dataset=cifar100

# Tiny-ImageNet, DINO pretraining
uv run python train.py --config-name tiny_imagenet_dino

# iNat18, DINO pretraining
uv run python train.py --config-name inat18_dino

# ImageNet, DINO pretraining
uv run python train.py --config-name imagenet_dino
```

All parameters are overridable from the CLI, e.g.
`architecture.model_type=resnet50 clustering.epochs=10`.

### Optional logging

Weights & Biases logging is enabled by default. Either set
`wandb.entity=<your-entity>` on the CLI or disable it with
`wandb.enabled=false` to fall back to CSV logging.

## Evaluation

Evaluation runs at the end of training and reports dendrogram purity,
hierarchical k-means / spectral clustering metrics, and kNN accuracy on the
held-out split. To evaluate frozen pretrained backbones with the same
metrics:

```bash
uv run python scripts/eval_baseline_embeddings.py --dataset cifar100 ...
```

## Repository layout

- `train.py` -- Hydra entry point, orchestrates the two-stage pipeline.
- `src/models/` -- Lightning modules (`HyperbolicResNet`,
  `DINOHyperbolicResNet`, `HyperbolicHCModel`, `HyperbolicAE`).
- `src/datasets/` -- datamodules and custom datasets.
- `src/hyperbolic_math/` -- manifolds (Euclidean, Hyperboloid, Poincare),
  Riemannian optimizers, hyperbolic neural-network layers.
- `src/evaluation.py` -- clustering metrics and 2D Poincare-ball
  visualization.
- `configs/` -- Hydra configs.
- `scripts/` -- utilities for dataset preparation and baseline evaluation.

## Tests

```bash
uv run pytest src/hyperbolic_math/tests/
```

## License

See `src/hyperbolic_math/LICENSE` for the hyperbolic math library license.
