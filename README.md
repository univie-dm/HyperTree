# HyperTree: Scalable Unsupervised Hierarchy Discovery in Hyperbolic Space

HyperTree is an unsupervised representation-learning method. It embeds images in hyperbolic space so that the
geometry of the embeddings encodes a hierarchy: similar images share deep lowest common ancestors (LCAs). To
evaluate the learned hierarchy, a tree is decoded from the embeddings by hierarchical clustering and compared with
the ground-truth class hierarchy (dendrogram purity). `train.py` runs two stages:

1. **Pretraining** (optional): contrastive (SimCLR) pretraining of a ResNet encoder and the Euclidean projection
   head.
2. **Joint training**: the encoder output passes through the projection head, feature scaling, the exponential map
   and a hyperbolic linear layer into the Poincaré ball (or the Hyperboloid). The embeddings are trained with the
   hierarchical loss, which is built on rectified hyperbolic LCAs and estimated from importance-sampled triplets,
   together with a contrastive (InfoNCE) loss on the geodesic distances.

Each stage evaluates at its start, every `experiment.log_interval` epochs, and after its last epoch. The evaluation
split is the test split of CIFAR-100 and the validation split of the other datasets.

## Setup

The hyperbolic math library [`hyperbolic-math`](https://doi.org/10.5281/zenodo.19236209) is a git submodule in
`src/hyperbolic_math`:

```bash
git clone --recurse-submodules https://github.com/univie-dm/HyperTree.git
cd HyperTree
uv sync --locked           # add --extra bioclip for the iNaturalist-2018 recipe (BioCLIP backbone)
```

`uv.lock` holds the exact environment that produced the paper results: every package and its version, including
Python 3.11, PyTorch 2.10 (CUDA 12.8) and PyTorch Lightning 2.6. `pyproject.toml` pins the direct dependencies.
Without uv, install those with pip into a Python 3.11 environment.
The commands below assume the environment is active (otherwise prefix them with `uv run`).

## Data

The dataset is selected by the recipe, and its location by `dataset.data_dir`:

| Dataset | `dataset.data_dir` (default) | Content |
|---|---|---|
| CIFAR-100 | `./data` | downloaded by torchvision on first use |
| TinyImageNet | `data/tiny-imagenet-200` | the extracted [`tiny-imagenet-200.zip`](http://cs231n.stanford.edu/tiny-imagenet-200.zip) |
| ImageNet-1k | `data/imagenet` | `ILSVRC2012_devkit_t12.tar.gz`, `ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar`, which torchvision's `ImageNet` extracts on first use |
| iNaturalist-2018 | `data/inat18` | the [official release](https://github.com/visipedia/inat_comp/tree/master/2018): `train2018.json`, `val2018.json` and the extracted `train_val2018/` |

The class hierarchies for hierarchical dendrogram purity are built automatically in `./data` on first use:
the WordNet hierarchies of CIFAR-100, TinyImageNet and ImageNet-1k, which need internet access once.
The iNaturalist-2018 taxonomy ships with the repository (`data/inat18_taxonomy_hierarchy.txt`).

## Reproducing the paper results

Each dataset has a recipe in `configs/`: `cifar100.yaml`, `tiny_imagenet.yaml`, `imagenet.yaml`, `inat18.yaml`.
These are the settings behind the main results (Appendix, "Hyperparameters for all main-paper experiments").
The paper reports seeds 1, 2 and 3 of each:

```bash
python train.py --config-name cifar100 experiment.seed=1
python train.py --config-name tiny_imagenet experiment.seed=1
python train.py --config-name imagenet experiment.seed=1
python train.py --config-name inat18 experiment.seed=1
```

The Hyperboloid rows of the main tables add `architecture.manifold_name=Hyperboloid`.

| Recipe | Backbone | Pretraining | Joint training | Measured wall time |
|---|---|---|---|---|
| CIFAR-100 | ResNet-50, trained | 800 epochs | 300 epochs, whole model | ~9.5 h (H100) |
| TinyImageNet | ResNet-50, trained | 800 epochs | 300 epochs, whole model | ~12.5 h (H100) |
| ImageNet-1k | DINO ResNet-50, frozen | none | 50 epochs, head only | ~35 h (H200) |
| iNaturalist-2018 | BioCLIP ViT-B/16 image encoder, frozen | none | 50 epochs, head only | ~20 h (H200) |

The backbone weights of ImageNet-1k and iNaturalist-2018 are downloaded on first use:

- **ImageNet-1k:** the public DINO ResNet-50, saved to `paths.models_dir`.
- **iNaturalist-2018:** BioCLIP (`hf-hub:imageomics/bioclip`), saved to the Hugging Face cache.

The wall times are those of one from-scratch run per recipe with this code, including all evaluations. With the
frozen backbones, they depend largely on how fast the images load from storage.
"Head only" means the projection head, feature scaling and hyperbolic layer. ImageNet-1k is evaluated on a fixed
class-stratified subsample of 10,000 validation images (`evaluation_metrics.test_sample_size`), the other datasets
on the full evaluation split.

The tables report the metrics of the final evaluation:

| Metric | Paper | Logged key |
|---|---|---|
| hDP | main results (Ward, LCA-induced tree), decoder ablation (single, average, complete, Ward) | `test/hierarchical_dendrogram_purity_<tree>_with_root`, where `<tree>` is `ward`, `complete`, `average`, `single` or `lca` (the LCA-induced tree) |
| Δ<sub>LCA</sub> | main results | LCA-induced hDP of the final evaluation minus that of the first joint-training evaluation (epoch 0) |
| 1-NN accuracy | appendix | `test/1_nearest_neighbors_accuracy` |
| NMI | appendix | `test/agglomerative_clustering_ward_nmi` |

The appendix ablations use seed 1 and these overrides:

| Ablation | Override |
|---|---|
| Contrastive only | `clustering.hierarchical_loss.start_weight=0 clustering.hierarchical_loss.end_weight=0` |
| Hierarchical only | `clustering.contrastive_loss.enabled=False` |
| Latent dimension | `architecture.latent_dim=<16, 32 or 128>` |
| Curvature (iNaturalist-2018) | `architecture.curvature_magnitude=<0.1 or 0.3>` |

The encoder and projection head run in float32 with PyTorch's `medium` float32 matmul precision, in training and
evaluation alike. The hyperbolic operations, including the geodesic distances of the evaluation, run in float64
(`architecture.manifold_precision`).
The embeddings lie close to the boundary of the ball, which makes the geodesic linkages sensitive to rounding.
On different hardware, Ward and complete-linkage hDP can therefore differ by a few 1e-3. The LCA-induced tree is
stable to about 1e-5. With the ResNet backbones, cuDNN's choice of convolution algorithm
(`experiment.cudnn_benchmark`) adds the same kind of noise between runs, even on the same GPU.

## Outputs

| Location | Default | Content |
|---|---|---|
| W&B project `wandb.project` | `hypertree` | all metrics; `wandb.enabled=False` writes them to a CSV logger in `wandb.save_dir` instead |
| `evaluation_metrics.results_dir` | `./results` | the metrics of every evaluation as JSON (`<stage>_results_latest.json`, `<stage>_results_history.jsonl`) |
| `paths.models_dir` | `./data/models` | the model at the end of pretraining, and downloaded backbone weights |
| `paths.checkpoints_dir` | `./data/checkpoints` | pretraining and joint-training checkpoints |
| `paths.final_models_dir` | `./data/final_models` | model and config after the last joint-training epoch |

The results files have fixed names, so give runs that execute at the same time different `results_dir`s.
To log to your own W&B account, set `wandb.entity`; it defaults to your default entity.

**Caching and resuming.**
- **Pretraining cache:** the pretrained model is stored under a hash of the pretraining settings. A run whose
  pretraining settings match a cached model skips pretraining and loads it; `pretraining.overwrite=True` forces
  a fresh pretraining.
- **Resuming:** joint-training checkpoints are keyed by a hash of the whole configuration. Starting the same
  configuration again resumes from the last checkpoint and continues the W&B run, e.g. after a Slurm time limit
  or preemption.
- **Path settings:** the `paths.*` settings do not enter either hash.

## Hardware

The paper runs used one NVIDIA H100 (80 GB) or H200 (141 GB) per run, 16 CPU cores, and 64 GB of host RAM (128 GB for
iNaturalist-2018).

- **CIFAR-100 and TinyImageNet:** both stages run at batch size 1024 (CIFAR-100) or 512 (TinyImageNet) with
  the encoder unfrozen, and need the 80 GB GPU.
- **ImageNet-1k and iNaturalist-2018:** the frozen backbones need less GPU memory.
- **Evaluation:** the linkage-based evaluation is bound by host RAM, since it needs dense pairwise distance
  matrices of the test split.

Overrides for smaller GPUs (these change the results):

| Override | Effect |
|---|---|
| `experiment.batch_size=128` | GPU memory scales with the batch size |
| `architecture.model_type=resnet18` | smaller encoder than the ResNet-50 |
| `experiment.training_precision=bf16-mixed` | mixed precision for the pretraining stage |
| `pretraining.accumulate_gradients=8` | recovers the effective pretraining batch size of a smaller `batch_size` |
| `clustering.freeze_encoder=True` | trains only the head in the joint-training stage |

A smoke test that runs both stages and the evaluation in a few minutes:

```bash
python train.py --config-name cifar100 wandb.enabled=False experiment.batch_size=128 \
  pretraining.epochs=1 clustering.epochs=2 evaluation_metrics.test_sample_size=2000
```

## Configuration

The configs use Hydra:

- **`configs/config.yaml`** holds the settings shared by all recipes. Loaded on its own, it is the CIFAR-100
  recipe.
- **`configs/<dataset>.yaml`** loads it and sets the values that differ per dataset: batch size, backbone and
  pretraining, projection head, embedding dimension, joint-training schedule, frozen backbone, and evaluation
  subsample.
- **`configs/dataset/`** defines the datasets.

Every option can be overridden on the command line, e.g. `python train.py --config-name imagenet
architecture.latent_dim=16 architecture.curvature_magnitude=0.3`. The options are documented in `config.yaml`.
The main ones:

**Backbone.**
- `pretraining.pretrained_weights=<path or URL>` loads an external ResNet checkpoint into the backbone and keeps
  it frozen. A URL is downloaded to `paths.models_dir`. Combine it with `pretraining.epochs=0`.
- `architecture.model_type=bioclip` uses the frozen BioCLIP image encoder. It needs the `bioclip` extra.

**Pretraining loss.** `pretraining.contrastive_loss.space`:
- `euclidean` (default): SimCLR on the projection-head features.
- `hyperbolic`: the contrastive loss on the geodesic distances of the embeddings.

**Trainable parts of the joint training.** The feature scaling and the hyperbolic layer are always trained.

| `clustering.freeze_encoder` | `clustering.train_projection_head` | Trained |
|---|---|---|
| `False` (CIFAR-100, TinyImageNet) | ignored | everything; an external or BioCLIP backbone stays frozen |
| `True` (ImageNet-1k, iNaturalist-2018) | `True` | projection head, feature scaling, hyperbolic layer |
| `True` | `False` | feature scaling, hyperbolic layer |

**Losses.** `L = λ_c L_contrastive + λ_h L_hierarchical`:
- **Contrastive loss:** its positive pair is the two augmented views of an image. `λ_c` is
  `clustering.contrastive_loss.weight` (0.1, constant).
- **Hierarchical loss schedule:** `λ_h` grows linearly from `start_weight` (1) to `end_weight` (5) over
  `weight_interval` epochs, and the soft-LCA temperature anneals from `start_temperature` (0.2) to
  `end_temperature` (0.1) over `temperature_interval` epochs (both 250 for CIFAR-100 and TinyImageNet, 50 for
  ImageNet-1k and iNaturalist-2018), all under `clustering.hierarchical_loss`.
- **Triplet sampling:** `sampling_eps` and `sampling_delta` (both 0.01) set the number of accepted triplets per
  step via the paper's sample-complexity bound (about 27,000).
- **Hierarchical loss:** computed on the first augmented view, reusing the pairwise distances of the contrastive
  loss.
- **Turning a loss off:** `clustering.contrastive_loss.enabled=False` trains with the hierarchical loss only.
  `clustering.hierarchical_loss.start_weight=0 clustering.hierarchical_loss.end_weight=0` trains with the
  contrastive loss only.

**Evaluation** (`evaluation_metrics`), every `experiment.log_interval` epochs on the test split, or on a fixed
class-stratified subsample of `test_sample_size` points:
- **Dendrogram purity:** hierarchical and flat, per tree. The trees are single, complete, average and Ward
  linkage on the geodesic distances, and the LCA-induced tree (single linkage with the pairwise rectified-LCA
  depths as similarities).
  Every purity is logged with the root merge (`_with_root`, as reported in the paper) and without it
  (`_no_root`).
- **NMI:** cuts a geodesic linkage tree into as many clusters as there are classes. The recipes use Ward, as in
  the paper; `agglomerative_clustering_<linkage>` enables the others.
- **k-NN accuracy:** majority vote of the k nearest neighbors under the geodesic distance.
- **Recall@K:** fraction of points that have at least one point of the same class among their K nearest
  neighbors (K = 2 and 10).

## License

MIT, see [`LICENSE`](LICENSE). The `hyperbolic-math` submodule is MIT-licensed as well.
