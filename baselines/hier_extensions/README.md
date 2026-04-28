# HIER paper-dataset extensions

The upstream HIER repo (https://github.com/clovaai/HIER-CVPR23) ships
adapters for CUB / Cars / SOP / Inshop only. This directory contains the
patches and new dataset adapters needed to run HIER on our paper
datasets (CIFAR-100, TinyImageNet, ImageNet) with the standard full-class
benchmark protocol used throughout `RESULTS.md`.

The cloned `baselines/HIER-CVPR23/` directory is gitignored, so these
files live here as the source of truth and get applied at job time.

## Files

- `dataset/cifar100.py` — full-class CIFAR-100 (50k train + 10k val, 100
  classes). Wraps `torchvision.datasets.CIFAR100`.
- `dataset/tiny_imagenet.py` — full-class TinyImageNet (100k train + 10k
  val, 200 classes). Wraps our `src.datasets.tiny_imagenet.TinyImageNet`.
- `dataset/imagenet.py` — full-class ImageNet-1k (1.28M + 50k val, 1000
  classes). Wraps `torchvision.datasets.ImageNet`.
- `dataset/__init__.py` — registers the new adapters in HIER's `_type`
  dict and re-exports them.
- `paper_datasets.patch` — single-file patch of the upstream HIER repo
  for `hier/train.py` (adds dataset choices + lifts the `epoch >= 40`
  eval gate for the new datasets) and `hier/dataset/__init__.py`. Apply
  with `git -C baselines/HIER-CVPR23 apply ../hier_extensions/paper_datasets.patch`.

## Apply after re-clone

```bash
cd baselines/HIER-CVPR23
cp ../hier_extensions/dataset/*.py hier/dataset/
git apply ../hier_extensions/paper_datasets.patch
```

Sbatches that train HIER on these datasets (`sbatch_cifar100_hier.sh`,
`sbatch_tiny_imagenet_hier.sh`, `sbatch_vsc_imagenet_hier.sh`) assume
this has been done.
