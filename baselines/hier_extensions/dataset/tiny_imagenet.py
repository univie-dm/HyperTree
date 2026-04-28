from __future__ import print_function
from __future__ import division

import sys
from pathlib import Path

import PIL.Image

from .base import BaseDataset

# Import our project's TinyImageNet adapter for the val-folder structure.
# Add repo root to sys.path so `src.datasets.tiny_imagenet` is importable when
# this file runs from inside baselines/HIER-CVPR23/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.datasets.tiny_imagenet import TinyImageNet as _TinyImageNet


class TinyImageNet(BaseDataset):
    """TinyImageNet adapter for HIER.

    Protocol: full-class split (all 200 classes in train and eval). HIER's
    native protocol is disjoint-class DML, but our paper compares on the
    standard TI benchmark (100k train + 10k val, all 200 classes) — same
    setup we use for the unsupervised clustering eval.
    """

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.classes = list(range(200))
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        split = "train" if mode == "train" else "val"
        ds_root = Path(root) / "tiny-imagenet-200"
        ds = _TinyImageNet(str(ds_root), split=split, transform=None)
        for idx, (img_path, label) in enumerate(ds.samples):
            self.ys.append(int(label))
            self.I.append(idx)
            self.im_paths.append(img_path)

    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        if len(list(im.split())) == 1:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, self.ys[index], index
