import argparse
import os
import torch
import pytorch_lightning as pl
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, DistributedSampler
from typing import Literal, Optional

# CIFAR-100 coarse label (20 superclasses) of each fine label
CIFAR100_COARSE_LABELS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
    3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
    0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
    16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
    2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
    18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


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
        if dataset_name in ("imagenet", "inat18"):
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
    A dataset that returns the requested views of an image together with its label,
    as a dict: 'original' (un-augmented base view), 'aug_1' and 'aug_2' (the two
    augmented views). Views that are not requested are not computed.
    """

    VIEWS = ("original", "aug_1", "aug_2")

    def __init__(self, dataset, base_transform, aug_transform: Optional[BYOLTransform] = None, views: tuple = ("original",)):
        self.dataset = dataset
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.views = tuple(views)
        if not set(self.views) <= set(self.VIEWS):
            raise ValueError(f"Invalid views: {self.views}. Supported views are: {self.VIEWS}.")
        if aug_transform is None and self.views != ("original",):
            raise ValueError("Augmented views require an augmentation transform.")
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
        sample = {"label": label}
        if "original" in self.views:
            sample["original"] = self.base_transform(img)
        if "aug_1" in self.views:
            sample["aug_1"] = self.aug_transform.transform1(img)
        if "aug_2" in self.views:
            sample["aug_2"] = self.aug_transform.transform2(img)
        return sample

    def __len__(self):
        return len(self.dataset)


class ColorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal[
            "cifar100", "imagenet", "inat18", "tiny_imagenet"
        ] = "cifar100",
        data_dir: str = "./data",
        batch_size: int = 256,
        resize: Optional[int] = 32,
        subset: str = "train",
        num_workers: int = 1,
        selected_classes: Optional[list] = None,
        precision: str = "float32",
        train_views: tuple = ("original", "aug_1", "aug_2"),
    ):
        super().__init__()
        self.hparams: argparse.Namespace
        self.batch_size = batch_size
        self.save_hyperparameters()
        self.dataset_map = {
            "cifar100": datasets.CIFAR100,
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

        # --- Base Datasets ---
        if self.hparams.dataset_name == "imagenet":
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
        # ImageNet is large and variable-sized so resize to self.hparams.resize if provided,
        # otherwise default to 224.
        if self.hparams.dataset_name in ("imagenet", "inat18", "tiny_imagenet"):
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
        else:
            # CIFAR and other torchvision datasets use `train=True` and support download
            temp_dataset = dataset_class(
                self.hparams.data_dir,
                train=True,
                transform=temp_transform,
                download=True,
            )

        # Use standard ImageNet normalization stats to avoid slow data loading
        if self.hparams.dataset_name in ("imagenet", "inat18", "tiny_imagenet"):
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
        aug_transform = BYOLTransform(
            dataset_name=self.hparams.dataset_name,
            normalize_fn=normalize_fn,
            resize=self.hparams.resize or 32,
            precision=self.precision,
        )

        # --- Filtering and Subsetting ---
        if self.hparams.selected_classes:
            train_mask = torch.tensor(
                [label in self.hparams.selected_classes for label in base_train.targets]
            )
            test_mask = torch.tensor(
                [label in self.hparams.selected_classes for label in base_test.targets]
            )
            if hasattr(base_train, "samples"):
                # ImageFolder-style datasets (iNat18, ImageNet): filter .samples
                # and .targets in lockstep; .imgs is an alias for .samples.
                train_keep = train_mask.tolist()
                test_keep = test_mask.tolist()
                base_train.samples = [s for s, k in zip(base_train.samples, train_keep) if k]
                base_train.imgs = base_train.samples
                base_test.samples = [s for s, k in zip(base_test.samples, test_keep) if k]
                base_test.imgs = base_test.samples
            else:
                base_train.data = base_train.data[train_mask]
                base_test.data = base_test.data[test_mask]
            base_train.targets = [
                label for i, label in enumerate(base_train.targets) if train_mask[i]
            ]
            base_test.targets = [
                label for i, label in enumerate(base_test.targets) if test_mask[i]
            ]

        # --- Create datasets based on subset ---
        # Training datasets only prepare the views the training stage consumes (train_views),
        # the evaluation dataset only the un-augmented original.
        train_views = tuple(self.hparams.train_views)
        if self.hparams.subset == "all":
            # When using 'all', both train and test data are used for training, so both get BYOL augmentations
            train_part = MultiViewDataset(base_train, base_transform, aug_transform, train_views)
            test_part_training = MultiViewDataset(
                base_test, base_transform, aug_transform, train_views
            )  # For training
            self.test_dataset = MultiViewDataset(base_test, base_transform)  # Clean for evaluation
            self.train_dataset = train_part  # Keep for consistency
            self.full_dataset = torch.utils.data.ConcatDataset(
                [train_part, test_part_training]
            )
        elif self.hparams.subset == "test":
            self.test_dataset = MultiViewDataset(base_test, base_transform)
            self.train_dataset = MultiViewDataset(
                base_test, base_transform, aug_transform, train_views
            )  # Use test set for training
            self.full_dataset = self.train_dataset
        elif self.hparams.subset == "train":
            self.train_dataset = MultiViewDataset(
                base_train, base_transform, aug_transform, train_views
            )
            self.test_dataset = MultiViewDataset(base_test, base_transform)
            self.full_dataset = self.train_dataset
        else:
            raise ValueError(f"Invalid subset: {self.hparams.subset}")

        self.ground_truth_labels = base_test.targets

    def train_dataloader(self) -> DataLoader:
        dataset = self.full_dataset
        
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
            # Bigger in-flight buffer so a momentary worker stall doesn't stall the GPU.
            prefetch_factor=4 if self.hparams.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        # No DistributedSampler: test_phase() runs only on rank 0 and needs the
        # full val set. Sharding it caused iNat18 to evaluate ~1 sample/class on
        # rank 0, producing degenerate DP=1, NMI=1, 1-NN=0 metrics.
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def get_coarse_labels(self) -> Optional[list[int]]:
        """Native coarse label of each fine label, for datasets that define one (CIFAR-100)."""
        if self.hparams.dataset_name == "cifar100":
            return CIFAR100_COARSE_LABELS
        return None
