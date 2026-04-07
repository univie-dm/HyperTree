"""
CUB-200-2011 dataset loader.

Handles the standard CUB-200-2011 folder structure:
- images/<class_folder>/*.jpg
- train_test_split.txt (image_id is_training)
- images.txt (image_id image_path)
- image_class_labels.txt (image_id class_id)

Download: https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
Extract into data/cub200/ so that data/cub200/images/ exists.
"""

import os
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class CUB200(Dataset):
    """
    CUB-200-2011 dataset (200 bird species, ~11.8k images).

    Parameters
    ----------
    root : str
        Root directory of the CUB-200-2011 dataset (containing images/, images.txt, etc.).
    split : str
        'train' or 'test'.
    transform : callable, optional
        Transform to apply to each image.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        if split not in ("train", "test"):
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'.")

        # Read metadata files
        images_txt = self.root / "images.txt"
        labels_txt = self.root / "image_class_labels.txt"
        split_txt = self.root / "train_test_split.txt"

        if not images_txt.exists():
            raise FileNotFoundError(
                f"CUB-200-2011 metadata not found at {self.root}. "
                "Download from https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz "
                "and extract so that images/, images.txt, etc. are in the root directory."
            )

        # Parse image paths: image_id -> relative_path
        image_paths = {}
        with open(images_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_paths[int(parts[0])] = parts[1]

        # Parse labels: image_id -> class_id (1-indexed in file, we convert to 0-indexed)
        image_labels = {}
        with open(labels_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_labels[int(parts[0])] = int(parts[1]) - 1  # 0-indexed

        # Parse train/test split: image_id -> is_training (1=train, 0=test)
        image_split = {}
        with open(split_txt) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    image_split[int(parts[0])] = int(parts[1])

        # Filter by split
        is_train = 1 if split == "train" else 0

        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []

        for img_id in sorted(image_paths.keys()):
            if image_split.get(img_id) != is_train:
                continue
            img_path = self.root / "images" / image_paths[img_id]
            label = image_labels[img_id]
            self.samples.append((str(img_path), label))
            self.targets.append(label)

        # Read class names
        classes_txt = self.root / "classes.txt"
        self.classes: list[str] = []
        if classes_txt.exists():
            with open(classes_txt) as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        self.classes.append(parts[1])

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.samples)
