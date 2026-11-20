"""
iNaturalist 2018 dataset loader.

Reads the official release as downloaded (https://github.com/visipedia/inat_comp/tree/master/2018):
- train2018.json, val2018.json: COCO-style annotations
- train_val2018/<supercategory>/<category_id>/*.jpg: the images
"""

import json
import os
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class INat2018(Dataset):
    """
    iNaturalist 2018 dataset (8142 species). The class index is the iNat category id, which is the leaf id of
    data/inat18_taxonomy_hierarchy.txt. Samples are ordered by class, then by file name.

    Parameters
    ----------
    root : str
        Root directory of the iNat2018 release (containing train2018.json, val2018.json, train_val2018/).
    split : str
        'train' or 'val'.
    transform : callable, optional
        Transform to apply to each image.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        if split not in ("train", "val"):
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'val'.")
        self.root = root
        self.split = split
        self.transform = transform

        with open(os.path.join(root, f"{split}2018.json")) as f:
            meta = json.load(f)
        self.classes = [str(c["id"]) for c in sorted(meta["categories"], key=lambda c: c["id"])]
        image_files = {image["id"]: image["file_name"] for image in meta["images"]}
        labelled_files = sorted(
            (annotation["category_id"], os.path.basename(image_files[annotation["image_id"]]), image_files[annotation["image_id"]])
            for annotation in meta["annotations"]
        )
        self.samples: list[tuple[str, int]] = [(os.path.join(root, file), label) for label, _, file in labelled_files]
        self.targets: list[int] = [label for _, label in self.samples]

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.samples)
