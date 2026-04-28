from __future__ import print_function
from __future__ import division

import torchvision
import PIL.Image

from .base import BaseDataset


class CIFAR100(BaseDataset):
    """
    CIFAR-100 adapter for HIER.

    Protocol: full-class split (all 100 classes in train and eval).
    This is a head-to-head comparison on the standard CIFAR-100 benchmark
    (50k train + 10k test, all 100 classes) against our unsupervised
    clustering method — NOT HIER's native disjoint-class DML protocol.

    Images are 32x32 uint8 PIL; the HIER transform pipeline resizes to
    224x224 (resize_size=256, crop_size=224) for the pretrained ResNet50.
    """

    def __init__(self, root, mode, transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.classes = range(0, 100)
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        train_flag = mode == "train"
        ds = torchvision.datasets.CIFAR100(
            root=self.root, train=train_flag, download=False
        )
        self._images = ds.data
        self._targets = list(ds.targets)

        for idx, y in enumerate(self._targets):
            self.ys.append(int(y))
            self.I.append(idx)
            self.im_paths.append(idx)

    def __getitem__(self, index):
        arr = self._images[index]
        im = PIL.Image.fromarray(arr)
        if self.transform is not None:
            im = self.transform(im)
        target = self.ys[index]
        return im, target, index
