# Deep Hyperbolic Hierarchical Clustering

Self-supervised representation learning with a hyperbolic clustering head.
Training is a two-stage pipeline orchestrated by `train.py`: contrastive/DINO
pretraining of an encoder, then hyperbolic hierarchical clustering on top of
the pretrained features.

## Setup

The hyperbolic math library is a submodule and must be present in `src/`:

```bash
git submodule update --init --recursive
uv sync
```

All commands run through `uv`:

```bash
uv run python train.py --config-name config dataset=cifar100
uv run pytest src/hyperbolic_math/tests/
```

## Hardware and system requirements

**A CUDA GPU is required.** The pretraining stage hardcodes
`accelerator="gpu"` (`train.py:139`), so there is no CPU fallback — on a
CPU-only machine the run fails rather than degrading gracefully. (Clustering
does have a CPU path, but it is impractically slow.)

Peak GPU memory, measured on CIFAR-100 for the four setups below. Figures are
`torch.cuda.max_memory_allocated()` during training steps, with reserved
memory in brackets:

| Setup | Pretraining | Clustering | Peak | Fits on |
|---|---|---|---|---|
| Smoke test | 0.63 GiB | 0.70 GiB | 0.70 [0.74] | any CUDA GPU |
| Reduced | 0.92 GiB | 0.78 GiB | 0.92 [1.02] | 4 GB+ |
| **Defaults** | **33.84 GiB** | 2.02 GiB | **33.84 [34.35]** | **40 GB+** |
| Paper (d=16) | 63.18 GiB | 4.04 GiB | 63.18 [64.16] | 80 GB |

Reproduce with `scripts/measure_memory.py`.

**Pretraining is the constraint, not clustering** — at the defaults it uses
roughly 17× more memory. The dominant term is `batch_size` × encoder
activations, so `batch_size` and `model_type` are the levers that matter.

Note the middle row: **`configs/config.yaml` as shipped needs a 40 GB+ GPU.**
It will not run on a 24 GB card.

The clustering column stays small on CIFAR-100 because the soft-LCA pairwise
distance matrix (`src/models/hyphc.py:393`) is O(B² × D) in float64, which is
negligible against conv activations at 32×32. This reverses at high
resolution: at 224² with ResNet50, clustering OOMs at `batch_size=256` on an
80 GB GPU while `batch_size=64` fits. For ImageNet- and iNat18-scale runs,
treat clustering as the binding constraint instead.

Other requirements:

| | Paper runs | Reduced | Smoke test |
|---|---|---|---|
| Host RAM | 64 GB | 32 GB | 16 GB |
| CPU cores | 16 | 8 | 4 |
| Wall time (CIFAR-100) | ~16 h | hours | minutes |
| Disk | ~5 GB (CIFAR-100); several hundred GB for ImageNet / iNat18 | | |

The measured figures cover training steps only. Evaluation runs separately
(`AbstractModel.on_train_start` and `on_train_end`) and its linkage and
spectral-clustering metrics are host-RAM bound rather than GPU bound, which
is what the RAM column reflects.

### Reducing memory

| Override | Effect |
|---|---|
| `experiment.batch_size=64` | Largest lever — pretraining memory scales with it directly. |
| `architecture.model_type=resnet18` | Much smaller encoder than the `resnet50` default. |
| `experiment.training_precision=bf16-mixed` | Autocast for the **pretraining stage only** (`train.py:135`). Hyperbolic ops stay FP32. |
| `pretraining.accumulate_gradients=8` | Recovers the effective batch size lost to a smaller `batch_size`. |
| `experiment.num_workers=4` | Reduces host RAM and CPU pressure, not VRAM. |
| `architecture.manifold_precision=float32` | Halves the pairwise-distance matrix. Only matters at high resolution; changes numerics, so not for reported results. |

## CIFAR-100 setups

### Smoke test (minutes, any GPU)

Verifies that both stages and the evaluation path run:

```bash
uv run python train.py --config-name config dataset=cifar100 \
  architecture.model_type=resnet18 \
  experiment.batch_size=64 \
  experiment.num_workers=4 \
  experiment.training_precision=bf16-mixed \
  pretraining.epochs=2 \
  clustering.epochs=2 \
  evaluation_metrics.test_sample_size=2000 \
  wandb.enabled=False
```

A configuration check, not a result.

### Reduced setup (hours, 4 GB+)

Trains something real at a fraction of the cost. Structure is visible;
absolute numbers will fall short of the paper.

```bash
uv run python train.py --config-name config dataset=cifar100 \
  architecture.model_type=resnet18 \
  architecture.latent_dim=16 \
  experiment.batch_size=128 \
  experiment.num_workers=8 \
  experiment.training_precision=bf16-mixed \
  pretraining.epochs=100 \
  pretraining.augmentation=byol \
  clustering.epochs=100 \
  clustering.freeze_encoder=True \
  wandb.enabled=False
```

### Paper run (~16 h on one 80 GB GPU)

**The defaults are not the paper recipe.** The published d=16 Poincaré
CIFAR-100 numbers come from this configuration, which differs from
`configs/config.yaml` in `batch_size`, `clustering.epochs`, `sampling_eps`
and `use_backbone_knn_similarities`:

```bash
uv run python train.py --config-name config dataset=cifar100 \
  experiment.seed=1 \
  experiment.batch_size=1024 \
  architecture.model_type=resnet50 \
  architecture.latent_dim=16 \
  architecture.manifold_name=PoincareBall \
  architecture.projection_layer_dims=[512,256,128] \
  pretraining.epochs=800 \
  pretraining.augmentation=byol \
  clustering.epochs=300 \
  clustering.freeze_encoder=True \
  clustering.hyperbolic_loss.sampling_eps=0.01 \
  clustering.hyperbolic_loss.use_backbone_knn_similarities=False
```

Run across seeds 1, 2, 3 and average.

### Pretraining is cached — reuse it

Pretrained encoders are saved to `data/models/{hash}.pt`, and a run whose
hash already exists **skips pretraining entirely** and loads the cached
weights (`train.py:175`). Since pretraining dominates both the ~16 h wall
time and the memory footprint, sweeping clustering knobs against a cached
encoder is dramatically cheaper — a few GiB of GPU memory instead of 60.

The hash covers the pretraining-relevant config (`src/utils/utils.py:22`),
including `seed`, `batch_size`, `optimizer` and the whole pretraining block,
so changing `batch_size` or the encoder invalidates it. Set
`pretraining.overwrite=True` to force a fresh encoder.

## Configuration

Hydra, with `configs/config.yaml` as the base and dataset overrides in
`configs/dataset/`. Every parameter is overridable from the CLI:

```bash
uv run python train.py --config-name config dataset=cifar100 architecture.latent_dim=16
uv run python train.py --config-name imagenet_dino
```

Top-level configs in `configs/` (`tiny_imagenet_simclr.yaml`,
`inat18_dino.yaml`, ...) are standalone and select their own dataset.

`wandb.enabled=False` swaps the W&B logger for a local CSV logger
(`src/utils/utils.py:307`), so no account is needed. With W&B enabled, set
`wandb.entity` to your own team.

## Cluster runs

Paper runs were submitted as single-GPU SLURM jobs with:

```bash
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
```

ImageNet- and iNat18-scale runs need more host RAM and considerably more disk.
