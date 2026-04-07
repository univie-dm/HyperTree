import argparse
import os
from pathlib import Path
import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, DistributedSampler
from typing import Literal, Optional, Dict, Tuple


class DINOTransform:
    """
    DINO-style multi-crop augmentation: 2 global crops (full resolution) + N local crops (smaller).
    Teacher sees only global crops; student sees all crops.
    Returns: (global1, global2, local1, ..., localN)
    """

    def __init__(
        self,
        dataset_name: str,
        normalize_fn: transforms.Normalize,
        resize: int = 224,
        precision: torch.dtype = torch.float32,
        n_local_crops: int = 6,
        local_crop_size: int = 96,
        global_scale: tuple = (0.4, 1.0),
        local_scale: tuple = (0.05, 0.4),
    ):
        self.n_local_crops = n_local_crops

        # Global crop augmentations (stronger)
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(resize, resize), scale=global_scale, antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23 if dataset_name in ("imagenet", "inat18", "cub200") else 3, sigma=(0.1, 2.0))
            ], p=1.0),  # Always blur first global crop
            transforms.ToImage(),
            transforms.ToDtype(precision, scale=True),
            normalize_fn,
        ])
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(resize, resize), scale=global_scale, antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23 if dataset_name in ("imagenet", "inat18", "cub200") else 3, sigma=(0.1, 2.0))
            ], p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToImage(),
            transforms.ToDtype(precision, scale=True),
            normalize_fn,
        ])

        # Local crop augmentations (smaller crops, lighter augmentation)
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=(local_crop_size, local_crop_size), scale=local_scale, antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23 if dataset_name in ("imagenet", "inat18", "cub200") else 3, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToImage(),
            transforms.ToDtype(precision, scale=True),
            normalize_fn,
        ])

    def __call__(self, img):
        global1 = self.global_transform1(img)
        global2 = self.global_transform2(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local_crops)]
        return (global1, global2, *locals_)


class BYOLTransform:
    """
    Applies two separate augmentation pipelines to an image, as required by BYOL.
    """

    def __init__(
        self,
        dataset_name: str,
        normalize_fn: transforms.Normalize,
        resize: int = 32,
        precision: torch.dtype = torch.float32,
    ):
        if dataset_name in ("imagenet", "inat18", "cub200"):
            gaussian_blur = [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1
                )
            ]
        elif dataset_name == "tiny_imagenet":
            gaussian_blur = [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.1
                )
            ]
        else:
            gaussian_blur = [
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.1
                )
            ]

        self.transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(resize, resize), scale=(0.2, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                *gaussian_blur,
                transforms.RandomSolarize(threshold=0.5, p=0.2),
                transforms.ToImage(),
                transforms.ToDtype(precision, scale=True),
                normalize_fn,
            ]
        )
        self.transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    size=(resize, resize), scale=(0.2, 1.0), antialias=True
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToImage(),
                transforms.ToDtype(precision, scale=True),
                normalize_fn,
            ]
        )

    def __call__(self, img):
        aug1 = self.transform1(img)
        aug2 = self.transform2(img)
        return aug1, aug2


class MultiViewDataset(torch.utils.data.Dataset):
    """
    A dataset that returns multiple views of an image: two augmented views
    for BYOL-style training and one base view for reconstruction.
    """

    def __init__(self, dataset, base_transform, aug_transform=None):
        self.dataset = dataset
        self.base_transform = base_transform
        self.aug_transform = (
            aug_transform if aug_transform is not None else base_transform
        )
        # Find all duplicate images in the dataset and remove them, keeping only the first occurrence
        # Work directly with the raw data (numpy array) instead of iterating over the dataset
        if hasattr(self.dataset, "data") and hasattr(self.dataset, "targets"):
            seen = set()
            unique_indices = []
            for idx in range(len(self.dataset.data)):
                img = self.dataset.data[idx]
                img_tuple = tuple(img.flatten().tolist())
                if img_tuple not in seen:
                    seen.add(img_tuple)
                    unique_indices.append(idx)
            self.dataset.data = self.dataset.data[unique_indices]
            self.dataset.targets = [self.dataset.targets[i] for i in unique_indices]

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        base_img = self.base_transform(img)

        if isinstance(self.aug_transform, DINOTransform):
            crops = self.aug_transform(img)
            # Return: (global1, global2, base_img, label, local1, ..., localN)
            return crops[0], crops[1], base_img, label, *crops[2:]
        elif isinstance(self.aug_transform, BYOLTransform):
            aug_img_1, aug_img_2 = self.aug_transform(img)
        else:
            # For non-BYOL transforms, use the same augmentation twice
            aug_img_1 = self.aug_transform(img)
            aug_img_2 = self.aug_transform(img)

        return aug_img_1, aug_img_2, base_img, label

    def __len__(self):
        return len(self.dataset)


class ColorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal[
            "cifar10", "cifar100", "cifar100_20", "gtsrb", "imagenet", "inat18", "tiny_imagenet", "cub200"
        ] = "cifar10",
        data_dir: str = "./data",
        batch_size: int = 256,
        resize: Optional[int] = 32,
        subset: str = "train",
        num_workers: int = 1,
        selected_classes: Optional[list] = None,
        precision: str = "float32",
        augmentation: str = "byol",  # 'byol' or 'dino'
        n_local_crops: int = 6,
        local_crop_size: int = 96,
        dino_global_scale: tuple = (0.4, 1.0),
        dino_local_scale: tuple = (0.05, 0.4),
    ):
        super().__init__()
        self.hparams: argparse.Namespace
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.dataset_map = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
            "cifar100_20": datasets.CIFAR100,
            "gtsrb": datasets.GTSRB,
            "imagenet": datasets.ImageNet,
        }
        self.precision: torch.dtype
        if precision == "float32":
            self.precision = torch.float32
        elif precision == "float64":
            self.precision = torch.float64
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    @staticmethod
    def get_normalize_fn(data: torch.Tensor) -> transforms.Normalize:
        channel_means = data.mean([0, 2, 3])
        channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(
            mean=channel_means.tolist(), std=channel_stds.tolist()
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_class = self.dataset_map.get(self.hparams.dataset_name)
        is_cifar100_20 = self.hparams.dataset_name == "cifar100_20"

        # --- Base Datasets ---
        if self.hparams.dataset_name == "gtsrb":
            base_train = dataset_class(
                self.hparams.data_dir, split="train", transform=None, download=True
            )
            base_test = dataset_class(
                self.hparams.data_dir, split="test", transform=None, download=True
            )
        elif self.hparams.dataset_name == "imagenet":
            base_train = dataset_class(
                self.hparams.data_dir, split="train", transform=None
            )
            base_test = dataset_class(
                self.hparams.data_dir, split="val", transform=None
            )
        elif self.hparams.dataset_name == "inat18":
            base_train = datasets.ImageFolder(
                os.path.join(self.hparams.data_dir, "train"), transform=None
            )
            base_test = datasets.ImageFolder(
                os.path.join(self.hparams.data_dir, "val"), transform=None
            )
        elif self.hparams.dataset_name == "tiny_imagenet":
            from src.datasets.tiny_imagenet import TinyImageNet
            base_train = TinyImageNet(self.hparams.data_dir, split="train", transform=None)
            base_test = TinyImageNet(self.hparams.data_dir, split="val", transform=None)
        elif self.hparams.dataset_name == "cub200":
            from src.datasets.cub200 import CUB200
            base_train = CUB200(self.hparams.data_dir, split="train", transform=None)
            base_test = CUB200(self.hparams.data_dir, split="test", transform=None)
        else:
            base_train = dataset_class(
                self.hparams.data_dir, train=True, transform=None, download=True
            )
            base_test = dataset_class(
                self.hparams.data_dir, train=False, transform=None, download=True
            )

        # --- Normalization ---
        temp_transform_list = []
        # Resize images for datasets with variable image sizes so tensors are stackable in the DataLoader.
        # GTSRB images are resized to 32x32 (keeps existing behavior). ImageNet is large and variable-sized
        # so resize to self.hparams.resize if provided, otherwise default to 224.
        if self.hparams.dataset_name == "gtsrb":
            temp_transform_list.append(transforms.Resize((32, 32), antialias=True))
        elif self.hparams.dataset_name in ("imagenet", "inat18", "tiny_imagenet", "cub200"):
            resize_size = self.hparams.resize if self.hparams.resize else 224
            temp_transform_list.append(
                transforms.Resize((resize_size, resize_size), antialias=True)
            )
        temp_transform_list.extend(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )
        temp_transform = transforms.Compose(temp_transform_list)
        # Create a temporary dataset used only to compute per-channel mean/std for normalization.
        # ImageNet's Dataset expects `split` and does not support `download` or `train` kwargs.
        if self.hparams.dataset_name == "imagenet":
            temp_dataset = dataset_class(
                self.hparams.data_dir, split="train", transform=temp_transform
            )
        elif self.hparams.dataset_name == "inat18":
            temp_dataset = datasets.ImageFolder(
                os.path.join(self.hparams.data_dir, "train"), transform=temp_transform
            )
        elif self.hparams.dataset_name == "tiny_imagenet":
            from src.datasets.tiny_imagenet import TinyImageNet
            temp_dataset = TinyImageNet(self.hparams.data_dir, split="train", transform=temp_transform)
        elif self.hparams.dataset_name == "cub200":
            from src.datasets.cub200 import CUB200
            temp_dataset = CUB200(self.hparams.data_dir, split="train", transform=temp_transform)
        elif self.hparams.dataset_name == "gtsrb":
            # GTSRB uses `split` and supports download
            temp_dataset = dataset_class(
                self.hparams.data_dir,
                split="train",
                transform=temp_transform,
                download=True,
            )
        else:
            # CIFAR and other torchvision datasets use `train=True` and support download
            temp_dataset = dataset_class(
                self.hparams.data_dir,
                train=True,
                transform=temp_transform,
                download=True,
            )

        # Use standard ImageNet normalization stats to avoid slow data loading
        if self.hparams.dataset_name in ("imagenet", "inat18", "tiny_imagenet", "cub200"):
            normalize_fn = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        else:
            temp_loader = DataLoader(
                temp_dataset, batch_size=1000, num_workers=self.hparams.num_workers
            )
            normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])

        # --- Transforms ---
        transform_list = []
        if self.hparams.resize:
            transform_list.append(
                transforms.Resize(
                    (self.hparams.resize, self.hparams.resize), antialias=True
                )
            )
        transform_list.extend(
            [
                transforms.ToImage(),
                transforms.ToDtype(self.precision, scale=True),
                normalize_fn,
            ]
        )
        base_transform = transforms.Compose(transform_list)

        # Always create augmentation transforms - let the dataset decide what to use
        if self.hparams.augmentation == "dino":
            aug_transform = DINOTransform(
                dataset_name=self.hparams.dataset_name,
                normalize_fn=normalize_fn,
                resize=self.hparams.resize or 224,
                precision=self.precision,
                n_local_crops=self.hparams.n_local_crops,
                local_crop_size=self.hparams.local_crop_size,
                global_scale=self.hparams.dino_global_scale,
                local_scale=self.hparams.dino_local_scale,
            )
        else:
            aug_transform = BYOLTransform(
                dataset_name=self.hparams.dataset_name,
                normalize_fn=normalize_fn,
                resize=self.hparams.resize or 32,
                precision=self.precision,
            )

        # --- Filtering and Subsetting ---
        if is_cifar100_20:
            base_train.targets = [self._get_coarse_label(t) for t in base_train.targets]
            base_test.targets = [self._get_coarse_label(t) for t in base_test.targets]

        if self.hparams.selected_classes:
            train_mask = torch.tensor(
                [label in self.hparams.selected_classes for label in base_train.targets]
            )
            test_mask = torch.tensor(
                [label in self.hparams.selected_classes for label in base_test.targets]
            )
            base_train.data = base_train.data[train_mask]
            base_train.targets = [
                label for i, label in enumerate(base_train.targets) if train_mask[i]
            ]
            base_test.data = base_test.data[test_mask]
            base_test.targets = [
                label for i, label in enumerate(base_test.targets) if test_mask[i]
            ]

        # --- Create datasets based on subset ---
        if self.hparams.subset == "all":
            # When using 'all', both train and test data are used for training, so both get BYOL augmentations
            train_part = MultiViewDataset(base_train, base_transform, aug_transform)
            test_part_training = MultiViewDataset(
                base_test, base_transform, aug_transform
            )  # For training
            self.test_dataset = MultiViewDataset(
                base_test, base_transform, base_transform
            )  # Clean for evaluation
            self.train_dataset = train_part  # Keep for consistency
            self.full_dataset = torch.utils.data.ConcatDataset(
                [train_part, test_part_training]
            )
        elif self.hparams.subset == "test":
            self.test_dataset = MultiViewDataset(
                base_test, base_transform, base_transform
            )
            self.train_dataset = MultiViewDataset(
                base_test, base_transform, aug_transform
            )  # Use test set for training
            self.full_dataset = self.test_dataset
        elif self.hparams.subset == "train":
            self.train_dataset = MultiViewDataset(
                base_train, base_transform, aug_transform
            )
            self.test_dataset = MultiViewDataset(
                base_test, base_transform, base_transform
            )
            self.full_dataset = self.train_dataset
        else:
            raise ValueError(f"Invalid subset: {self.hparams.subset}")

        self.ground_truth_labels = base_test.targets

    def train_dataloader(self) -> DataLoader:
        dataset = {
            "all": self.full_dataset,
            "test": self.test_dataset,
            "train": self.train_dataset,
        }[self.hparams.subset]
        
        # Check if we are in a distributed environment
        # self.trainer.world_size is the Lightning way to check this
        is_distributed = self.trainer is not None and self.trainer.world_size > 1
        
        sampler = DistributedSampler(dataset) if is_distributed else None
        
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.hparams.batch_size,
            # CRITICAL: shuffle must be False if using a sampler
            shuffle=True if sampler is None else False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        # Use the correct dataset for testing
        dataset = self.test_dataset
        
        is_distributed = self.trainer is not None and self.trainer.world_size > 1
        
        # We use a sampler here too so each GPU validates a different chunk of data
        sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None
        
        return DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )
    @staticmethod
    def _get_coarse_label(fine_label: int) -> int:
        """Convert CIFAR100 fine labels to coarse labels (20 classes)"""
        _coarse_labels = [
            4,
            1,
            14,
            8,
            0,
            6,
            7,
            7,
            18,
            3,
            3,
            14,
            9,
            18,
            7,
            11,
            3,
            9,
            7,
            11,
            6,
            11,
            5,
            10,
            7,
            6,
            13,
            15,
            3,
            15,
            0,
            11,
            1,
            10,
            12,
            14,
            16,
            9,
            11,
            5,
            5,
            19,
            8,
            8,
            15,
            13,
            14,
            17,
            18,
            10,
            16,
            4,
            17,
            4,
            2,
            0,
            17,
            4,
            18,
            17,
            10,
            3,
            2,
            12,
            12,
            16,
            12,
            1,
            9,
            19,
            2,
            10,
            0,
            1,
            16,
            12,
            9,
            13,
            15,
            13,
            16,
            19,
            2,
            4,
            6,
            19,
            5,
            5,
            8,
            19,
            18,
            1,
            2,
            15,
            6,
            0,
            17,
            8,
            14,
            13,
        ]
        return _coarse_labels[fine_label]

    def get_class_names(self) -> Optional[Dict[int, Tuple[str, Optional[str]]]]:
        """Return class names for the dataset"""
        if self.hparams.dataset_name == "cifar10":
            fine_labels = datasets.CIFAR10(self.hparams.data_dir, train=False).classes
            return {i: (fine_labels[i], None) for i in range(len(fine_labels))}
        elif self.hparams.dataset_name in ["cifar100", "cifar100_20"]:
            # CIFAR-100 coarse labels (20 superclasses)
            coarse_labels = [
                "aquatic_mammals",
                "fish",
                "flowers",
                "food_containers",
                "fruit_and_vegetables",
                "household_electrical_devices",
                "household_furniture",
                "insects",
                "large_carnivores",
                "large_man-made_outdoor_things",
                "large_natural_outdoor_scenes",
                "large_omnivores_and_herbivores",
                "medium_mammals",
                "non-insect_invertebrates",
                "people",
                "reptiles",
                "small_mammals",
                "trees",
                "vehicles_1",
                "vehicles_2",
            ]

            # For cifar100 (fine labels), map each fine label to its coarse label
            if self.hparams.dataset_name == "cifar100":
                fine_labels = datasets.CIFAR100(
                    self.hparams.data_dir, train=False
                ).classes
                return {
                    i: (fine_labels[i], coarse_labels[self._get_coarse_label(i)])
                    for i in range(100)
                }
            else:  # cifar100_20
                # For cifar100_20, the labels are already coarse (0-19)
                return {i: (coarse_labels[i], None) for i in range(20)}
        elif self.hparams.dataset_name in ("imagenet", "tiny_imagenet"):
            return self._get_imagenet_class_names()
        elif self.hparams.dataset_name == "gtsrb":
            # Implementation for GTSRB class names would go here
            # Simplified for brevity
            return None
        else:
            return None

    def _get_imagenet_class_names(self) -> Optional[Dict[int, Tuple[str, Optional[str]]]]:
        """Load ImageNet class names from the cached class index JSON.

        For Tiny ImageNet, maps the dataset's local indices (0-199) to human-readable
        names using the synset IDs from the dataset.
        """
        import json
        cache_path = Path("./data/imagenet_class_index.json")
        if not cache_path.exists():
            try:
                import requests
                url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
                resp = requests.get(url)
                resp.raise_for_status()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(resp.text)
            except Exception:
                return None
        try:
            class_index = json.loads(cache_path.read_text())
            # class_index: {"0": ["n01440764", "tench"], "1": ["n01443537", "goldfish"], ...}
            # Build synset -> name mapping from the full ImageNet class index
            synset_to_name = {v[0]: v[1].replace("_", " ") for v in class_index.values()}

            if self.hparams.dataset_name == "tiny_imagenet":
                # Map Tiny ImageNet's local indices to names using synset IDs
                from src.datasets.tiny_imagenet import TinyImageNet
                ds = TinyImageNet(self.hparams.data_dir, split="train")
                return {
                    idx: (synset_to_name.get(synset, synset), None)
                    for idx, synset in enumerate(ds.classes)
                }
            else:
                return {int(k): (v[1].replace("_", " "), None) for k, v in class_index.items()}
        except Exception:
            return None


class GrayScaleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal["mnist", "fmnist", "kmnist", "usps"] = "mnist",
        data_dir: str = "./data",
        batch_size: int = 256,
        resize: int = 28,
        subset: str = "train",
        num_workers: int = 1,
        selected_classes: Optional[list] = None,
    ):
        super().__init__()
        self.hparams: argparse.Namespace
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.dataset_map = {
            "mnist": datasets.MNIST,
            "fmnist": datasets.FashionMNIST,
            "kmnist": datasets.KMNIST,
            "usps": datasets.USPS,
        }

    @staticmethod
    def get_normalize_fn(data: torch.Tensor) -> transforms.Normalize:
        channel_means = data.mean([0, 2, 3])
        channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(
            mean=channel_means.tolist(), std=channel_stds.tolist()
        )

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_class = self.dataset_map[self.hparams.dataset_name]

        # First download the dataset
        temp_transform = transforms.Compose(
            [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
        )
        dataset_class(
            self.hparams.data_dir, train=False, transform=temp_transform, download=True
        )
        temp_dataset = dataset_class(
            self.hparams.data_dir, train=True, transform=temp_transform, download=True
        )
        temp_loader = DataLoader(
            temp_dataset, batch_size=1000, num_workers=self.hparams.num_workers
        )
        normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])

        # Create transforms
        transform_list = []
        if self.hparams.resize:
            transform_list.append(
                transforms.Resize(
                    (self.hparams.resize, self.hparams.resize), antialias=True
                )
            )
        transform_list.extend(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                normalize_fn,
            ]
        )
        base_transform = transforms.Compose(transform_list)

        # Always create augmentation transforms - let the dataset decide what to use
        aug_transform_list = []
        if self.hparams.resize:
            aug_transform_list.append(
                transforms.Resize(
                    (self.hparams.resize, self.hparams.resize), antialias=True
                )
            )
        aug_transform_list.extend(
            [
                transforms.RandomAffine(
                    degrees=(-16.0, 16.0), translate=(0.1, 0.1), shear=(-8.0, 8.0)
                ),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                normalize_fn,
            ]
        )
        aug_transform = transforms.Compose(aug_transform_list)

        # Load base datasets
        base_test = dataset_class(self.hparams.data_dir, train=False, transform=None)
        base_train = dataset_class(self.hparams.data_dir, train=True, transform=None)

        # Filter for selected classes if specified
        if self.hparams.selected_classes is not None:
            # Validate selected classes
            max_class = max(base_train.targets)
            invalid_classes = [
                c for c in self.hparams.selected_classes if c > max_class
            ]
            if invalid_classes:
                raise ValueError(
                    f"Invalid class labels for {self.hparams.dataset_name}: {invalid_classes}. "
                    f"Maximum class label is {max_class}."
                )

            # Create masks for selected classes
            train_mask = torch.tensor(
                [label in self.hparams.selected_classes for label in base_train.targets]
            )
            test_mask = torch.tensor(
                [label in self.hparams.selected_classes for label in base_test.targets]
            )

            # Filter datasets
            base_train.data = base_train.data[train_mask]
            base_train.targets = [
                label for i, label in enumerate(base_train.targets) if train_mask[i]
            ]
            base_test.data = base_test.data[test_mask]
            base_test.targets = [
                label for i, label in enumerate(base_test.targets) if test_mask[i]
            ]

        # Create datasets based on subset
        if self.hparams.subset == "all":
            self.train_dataset = MultiViewDataset(
                base_train, base_transform, aug_transform
            )
            self.test_dataset = MultiViewDataset(
                base_test, base_transform, base_transform
            )
            self.full_dataset = torch.utils.data.ConcatDataset(
                [self.train_dataset, self.test_dataset]
            )
        elif self.hparams.subset == "smallest":
            self.train_dataset = MultiViewDataset(
                base_test, base_transform, aug_transform
            )
            self.test_dataset = MultiViewDataset(
                base_test, base_transform, base_transform
            )
            self.full_dataset = self.test_dataset
        elif self.hparams.subset == "train":
            self.train_dataset = MultiViewDataset(
                base_train, base_transform, aug_transform
            )
            self.test_dataset = MultiViewDataset(
                base_test, base_transform, base_transform
            )
            self.full_dataset = self.train_dataset
        else:
            raise ValueError(f"Invalid subset: {self.hparams.subset}")

        self.ground_truth_labels = base_test.targets

    def train_dataloader(self) -> DataLoader:
        if self.hparams.subset == "all":
            dataset = self.full_dataset
        elif self.hparams.subset == "smallest":
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset

        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def get_class_names(self) -> Optional[Dict[int, Tuple[str, Optional[str]]]]:
        """Return class names for the dataset.

        Returns a dictionary mapping label IDs to tuples of (fine_label, coarse_label).
        For flat datasets, coarse_label is None.
        """
        if self.hparams.dataset_name == "mnist":
            # MNIST has flat labels (no hierarchy)
            return {
                0: ("0-Zero", None),
                1: ("1-One", None),
                2: ("2-Two", None),
                3: ("3-Three", None),
                4: ("4-Four", None),
                5: ("5-Five", None),
                6: ("6-Six", None),
                7: ("7-Seven", None),
                8: ("8-Eight", None),
                9: ("9-Nine", None),
            }
        elif self.hparams.dataset_name == "fmnist":
            # Fashion-MNIST has hierarchical labels
            # Coarse categories: Tops, Bottoms, Footwear, Accessories
            return {
                0: ("T-shirt/top", "Tops"),
                1: ("Trouser", "Bottoms"),
                2: ("Pullover", "Tops"),
                3: ("Dress", "Bottoms"),
                4: ("Coat", "Tops"),
                5: ("Sandal", "Footwear"),
                6: ("Shirt", "Tops"),
                7: ("Sneaker", "Footwear"),
                8: ("Bag", "Accessories"),
                9: ("Ankle boot", "Footwear"),
            }
        elif self.hparams.dataset_name == "kmnist":
            # KMNIST has flat labels (Japanese characters, no natural hierarchy)
            return {
                0: ("o", None),
                1: ("ki", None),
                2: ("su", None),
                3: ("tsu", None),
                4: ("na", None),
                5: ("ha", None),
                6: ("ma", None),
                7: ("ya", None),
                8: ("re", None),
                9: ("wo", None),
            }
        else:
            return None


class TabularMultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None, aug_transform=None):
        self.data = torch.FloatTensor(data)
        self.targets = torch.LongTensor(targets)
        self.transform = transform
        self.aug_transform = aug_transform if aug_transform is not None else transform

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        # Apply basic transform
        if self.transform:
            x_base = self.transform(x)
        else:
            x_base = x

        # Apply augmentation transform
        if self.aug_transform:
            x_aug = self.aug_transform(x)
        else:
            x_aug = x_base

        return x_aug, x_base, y

    def __len__(self):
        return len(self.data)


class TabularTransform:
    """Basic transform for tabular data"""

    def __init__(self, noise_std: float = 0.0):
        self.noise_std = noise_std

    def __call__(self, x):
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


class TabularDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal["iris"] = "iris",
        batch_size: int = 32,
        subset: str = "train",
        num_workers: int = 1,
        train_ratio: float = 0.8,
        aug_noise_std: float = 0.1,
    ):
        super().__init__()
        self.hparams: argparse.Namespace
        self.save_hyperparameters()
        self.dataset_map = {"iris": load_iris}

    def prepare_data(self) -> None:
        # Download datasets if needed (not needed for iris)
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Load dataset
        dataset_fn = self.dataset_map[self.hparams.dataset_name]
        dataset = dataset_fn()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dataset.data)  # type: ignore

        # Split into train/test
        num_samples = len(dataset.data)  # type: ignore
        num_train = int(num_samples * self.hparams.train_ratio)
        indices = np.random.permutation(num_samples)
        train_idx, test_idx = indices[:num_train], indices[num_train:]

        # Create transforms
        base_transform = TabularTransform(noise_std=0.0)
        aug_transform = TabularTransform(noise_std=self.hparams.aug_noise_std)

        # Create train/test datasets
        if self.hparams.subset == "all":
            self.train_dataset = TabularMultiViewDataset(
                X_scaled[train_idx],
                dataset.target[train_idx],  # type: ignore
                transform=base_transform,
                aug_transform=base_transform,
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx],
                dataset.target[test_idx],  # type: ignore
                transform=base_transform,
                aug_transform=base_transform,
            )
        else:
            self.train_dataset = TabularMultiViewDataset(
                X_scaled[train_idx],
                dataset.target[train_idx],  # type: ignore
                transform=base_transform,
                aug_transform=aug_transform,
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx],
                dataset.target[test_idx],  # type: ignore
                transform=base_transform,
                aug_transform=base_transform,
            )

        self.full_dataset = torch.utils.data.ConcatDataset(
            [self.train_dataset, self.test_dataset]
        )
        self.ground_truth_labels = dataset.target  # type: ignore
        self.input_dim = dataset.data.shape[1]  # Number of features  # type: ignore

    def train_dataloader(self) -> DataLoader:
        if self.hparams.subset == "all":
            dataset = self.full_dataset
        else:
            dataset = self.train_dataset
        return DataLoader(
            dataset,  # type: ignore
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def get_class_names(self) -> Dict[int, str]:
        return {0: "Setosa", 1: "Versicolour", 2: "Virginica"}
