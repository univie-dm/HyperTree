from __future__ import print_function
from __future__ import division

from pathlib import Path

import PIL.Image
import torchvision.datasets as tv_datasets

from .base import BaseDataset


class ImageNet(BaseDataset):
    """ImageNet-1k adapter for HIER.

    Protocol: full-class split (all 1000 classes in train and eval). HIER's
    native DML protocol is disjoint-class; here we run the standard
    benchmark (1.28M train + 50k val, all 1000 classes) for matched
    comparison with our pipeline.

    Expects torchvision's ImageNet-style folder layout under
    `<root>/imagenet/{train,val}/<synset>/<image>.JPEG`. Devkit tarball
    (`ILSVRC2012_devkit_t12.tar.gz`) must be in `<root>/imagenet/` for
    torchvision.datasets.ImageNet.
    """

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.classes = list(range(1000))
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        split = "train" if mode == "train" else "val"
        ds = tv_datasets.ImageNet(
            root=str(Path(root) / "imagenet"), split=split, transform=None,
        )
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
