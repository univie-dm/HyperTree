"""
Tiny ImageNet dataset loader.

Handles the non-standard folder structure:
- train: train/<synset>/images/*.JPEG
- val: val/images/*.JPEG + val/val_annotations.txt
"""

import os
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    """
    Tiny ImageNet dataset (200 classes, 64×64 images).

    Parameters
    ----------
    root : str
        Root directory of the Tiny ImageNet dataset (containing train/, val/, wnids.txt).
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
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Read the 200 class synset IDs (sorted for consistent indexing)
        wnids_file = self.root / "wnids.txt"
        if wnids_file.exists():
            with open(wnids_file) as f:
                self.classes = sorted(line.strip() for line in f if line.strip())
        else:
            # Fall back to reading train directory names
            self.classes = sorted(
                d.name
                for d in (self.root / "train").iterdir()
                if d.is_dir() and d.name.startswith("n")
            )

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []

        if split == "train":
            self._load_train()
        elif split == "val":
            self._load_val()
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'val'.")

    def _load_train(self):
        """Load training data from train/<synset>/images/*.JPEG."""
        train_dir = self.root / "train"
        for cls in self.classes:
            cls_img_dir = train_dir / cls / "images"
            if not cls_img_dir.exists():
                continue
            label = self.class_to_idx[cls]
            for img_path in sorted(cls_img_dir.iterdir()):
                if img_path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    self.samples.append((str(img_path), label))
                    self.targets.append(label)

    def _load_val(self):
        """Load validation data using val_annotations.txt."""
        val_dir = self.root / "val"
        annotations_file = val_dir / "val_annotations.txt"

        if annotations_file.exists():
            with open(annotations_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        fname, synset = parts[0], parts[1]
                        if synset in self.class_to_idx:
                            img_path = val_dir / "images" / fname
                            if img_path.exists():
                                label = self.class_to_idx[synset]
                                self.samples.append((str(img_path), label))
                                self.targets.append(label)
        else:
            # Fallback: val organized as val/<synset>/images/*.JPEG
            for cls in self.classes:
                cls_img_dir = val_dir / cls / "images"
                if not cls_img_dir.exists():
                    continue
                label = self.class_to_idx[cls]
                for img_path in sorted(cls_img_dir.iterdir()):
                    if img_path.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        self.samples.append((str(img_path), label))
                        self.targets.append(label)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self) -> int:
        return len(self.samples)
