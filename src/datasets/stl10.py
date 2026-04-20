"""
STL10 wrapper exposing a ``targets`` list (in sync with ``labels``) so the
dataset works with the shared filtering / dedup code in ``datamodule.py``.
"""

from typing import Callable, Optional

import numpy as np
from torchvision import datasets


class STL10(datasets.STL10):
    @property
    def targets(self) -> list:
        return self.labels.tolist()

    @targets.setter
    def targets(self, value) -> None:
        self.labels = np.asarray(value, dtype=np.int64)

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        super().__init__(root=root, split=split, transform=transform, download=download)
