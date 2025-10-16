import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from typing import Literal, Optional, Union, Dict, Tuple


class BYOLTransform:
    """
    Applies two separate augmentation pipelines to an image, as required by BYOL.
    """
    def __init__(self, normalize_fn: transforms.Normalize, resize: int = 32):
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(size=resize, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.1),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToTensor(),
            normalize_fn
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(size=resize, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize_fn
        ])

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
        self.aug_transform = aug_transform if aug_transform is not None else base_transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        base_img = self.base_transform(img)
        aug_img_1, aug_img_2 = self.aug_transform(img)
        return aug_img_1, aug_img_2, base_img, label


    def __len__(self):
        return len(self.dataset)


class ColorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal["cifar10", "cifar100", "cifar100_20", "gtsrb", "imagenet"] = "cifar10",
        data_dir: str = './data',
        batch_size: int = 256,
        resize: Optional[int] = 32,
        subset: str = 'train',
        num_workers: int = 1,
        selected_classes: Optional[list] = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_map = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
            "cifar100_20": datasets.CIFAR100,
            "gtsrb": datasets.GTSRB,
            "imagenet": datasets.ImageNet,
        }

    @staticmethod
    def get_normalize_fn(data: torch.Tensor) -> transforms.Normalize:
        channel_means = data.mean([0, 2, 3])
        channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(channel_means, channel_stds)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_class = self.dataset_map[self.hparams.dataset_name]
        is_cifar100_20 = self.hparams.dataset_name == "cifar100_20"

        # --- Base Datasets ---
        if self.hparams.dataset_name == "gtsrb":
            base_train = dataset_class(self.hparams.data_dir, split='train', transform=None, download=True)
            base_test = dataset_class(self.hparams.data_dir, split='test', transform=None, download=True)
        else:
            base_train = dataset_class(self.hparams.data_dir, train=True, transform=None, download=True)
            base_test = dataset_class(self.hparams.data_dir, train=False, transform=None, download=True)

        # --- Normalization ---
        temp_transform = transforms.Compose([
            transforms.Resize((32, 32)) if self.hparams.dataset_name == "gtsrb" else transforms.Identity(),
            transforms.ToTensor()
        ])
        temp_dataset = dataset_class(
            self.hparams.data_dir, 
            **({"train": True} if self.hparams.dataset_name != "gtsrb" else {"split": "train"}),
            transform=temp_transform, 
            download=True
        )
        temp_loader = DataLoader(temp_dataset, batch_size=1000, num_workers=self.hparams.num_workers)
        normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])

        # --- Transforms ---
        transform_list = []
        if self.hparams.resize:
            transform_list.append(transforms.Resize((self.hparams.resize, self.hparams.resize)))
        transform_list.extend([transforms.ToTensor(), normalize_fn])
        base_transform = transforms.Compose(transform_list)

        # Always create augmentation transforms - let the dataset decide what to use
        aug_transform = BYOLTransform(normalize_fn, resize=self.hparams.resize or 32)

        # --- Filtering and Subsetting ---
        if is_cifar100_20:
            base_train.targets = [self._get_coarse_label(t) for t in base_train.targets]
            base_test.targets = [self._get_coarse_label(t) for t in base_test.targets]

        if self.hparams.selected_classes:
            train_mask = torch.tensor([label in self.hparams.selected_classes for label in base_train.targets])
            test_mask = torch.tensor([label in self.hparams.selected_classes for label in base_test.targets])
            base_train.data = base_train.data[train_mask]
            base_train.targets = [label for i, label in enumerate(base_train.targets) if train_mask[i]]
            base_test.data = base_test.data[test_mask]
            base_test.targets = [label for i, label in enumerate(base_test.targets) if test_mask[i]]

        # --- Create datasets based on subset ---
        if self.hparams.subset == 'all':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        elif self.hparams.subset == 'test':
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.train_dataset = MultiViewDataset(base_test, base_transform, aug_transform)  # Use test set for training
            self.full_dataset = self.test_dataset
        elif self.hparams.subset == 'train':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = self.train_dataset
        else:
            raise ValueError(f"Invalid subset: {self.hparams.subset}")

        self.ground_truth_labels = base_test.targets

    def train_dataloader(self) -> DataLoader:
        dataset = {'all': self.full_dataset, 'test': self.test_dataset, 'train': self.train_dataset}[self.hparams.subset]
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )

    @staticmethod
    def _get_coarse_label(fine_label: int) -> int:
        """Convert CIFAR100 fine labels to coarse labels (20 classes)"""
        _coarse_labels = [4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                         3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                         6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                         0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                         5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                         16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                         10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                         2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                         16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                         18, 1, 2, 15, 6, 0, 17, 8, 14, 13]
        return _coarse_labels[fine_label]

    def get_class_names(self) -> Optional[Dict[int, Tuple[str, Optional[str]]]]:
        """Return class names for the dataset"""
        if self.hparams.dataset_name == "cifar10":
            fine_labels = datasets.CIFAR10(self.hparams.data_dir, train=False).classes
            return {i: (fine_labels[i], None) for i in range(len(fine_labels))}
        elif self.hparams.dataset_name in ["cifar100", "cifar100_20"]:
            # Implementation for CIFAR100 class names would go here
            # Simplified for brevity
            return None
        elif self.hparams.dataset_name == "gtsrb":
            # Implementation for GTSRB class names would go here
            # Simplified for brevity
            return None
        else:
            return None


class GrayScaleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: Literal["mnist", "fmnist", "kmnist", "usps"] = "mnist",
        data_dir: str = './data',
        batch_size: int = 256,
        resize: int = 28,
        subset: str = 'train',
        num_workers: int = 1,
        selected_classes: Optional[list] = None
    ):
        super().__init__()
        self.save_hyperparameters()
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
        return transforms.Normalize(channel_means, channel_stds)

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_class = self.dataset_map[self.hparams.dataset_name]

        # First download the dataset
        dataset_class(self.hparams.data_dir, train=False, transform=transforms.ToTensor(), download=True)
        temp_dataset = dataset_class(self.hparams.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        temp_loader = DataLoader(temp_dataset, batch_size=1000, num_workers=self.hparams.num_workers)
        normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])

        # Create transforms
        transform_list = []
        if self.hparams.resize:
            transform_list.append(transforms.Resize((self.hparams.resize, self.hparams.resize)))
        transform_list.extend([transforms.ToTensor(), normalize_fn])
        base_transform = transforms.Compose(transform_list)

        # Always create augmentation transforms - let the dataset decide what to use
        aug_transform = transforms.Compose([
            *transform_list[:-2],  # All except ToTensor and normalize
            transforms.RandomAffine(degrees=16, translate=(0.1, 0.1), shear=8),
            transforms.ToTensor(),
            normalize_fn
        ])

        # Load base datasets
        base_test = dataset_class(self.hparams.data_dir, train=False, transform=None)
        base_train = dataset_class(self.hparams.data_dir, train=True, transform=None)

        # Filter for selected classes if specified
        if self.hparams.selected_classes is not None:
            # Validate selected classes
            max_class = max(base_train.targets)
            invalid_classes = [c for c in self.hparams.selected_classes if c > max_class]
            if invalid_classes:
                raise ValueError(f"Invalid class labels for {self.hparams.dataset_name}: {invalid_classes}. "
                               f"Maximum class label is {max_class}.")

            # Create masks for selected classes
            train_mask = torch.tensor([label in self.hparams.selected_classes for label in base_train.targets])
            test_mask = torch.tensor([label in self.hparams.selected_classes for label in base_test.targets])

            # Filter datasets
            base_train.data = base_train.data[train_mask]
            base_train.targets = [label for i, label in enumerate(base_train.targets) if train_mask[i]]
            base_test.data = base_test.data[test_mask]
            base_test.targets = [label for i, label in enumerate(base_test.targets) if test_mask[i]]

        # Create datasets based on subset
        if self.hparams.subset == 'all':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        elif self.hparams.subset == 'smallest':
            self.train_dataset = MultiViewDataset(base_test, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = self.test_dataset
        elif self.hparams.subset == 'train':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = self.train_dataset
        else:
            raise ValueError(f"Invalid subset: {self.hparams.subset}")

        self.ground_truth_labels = base_test.targets

    def train_dataloader(self) -> DataLoader:
        if self.hparams.subset == 'all':
            dataset = self.full_dataset
        elif self.hparams.subset == 'smallest':
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset
        
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True,
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers, 
            pin_memory=True
        )

    def get_class_names(self) -> Optional[Dict[int, str]]:
        if self.hparams.dataset_name == "mnist":
            return {
                0: "0-Zero", 1: "1-One", 2: "2-Two", 3: "3-Three", 4: "4-Four",
                5: "5-Five", 6: "6-Six", 7: "7-Seven", 8: "8-Eight", 9: "9-Nine"
            }
        elif self.hparams.dataset_name == "fmnist":
            return {
                0: "0-T-shirt/top", 1: "1-Trouser", 2: "2-Pullover", 3: "3-Dress", 4: "4-Coat",
                5: "5-Sandal", 6: "6-Shirt", 7: "7-Sneaker", 8: "8-Bag", 9: "9-Ankle boot"
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
        subset: str = 'train',
        num_workers: int = 1,
        train_ratio: float = 0.8,
        aug_noise_std: float = 0.1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_map = {
            "iris": load_iris
        }

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
        if self.hparams.subset == 'all':
            self.train_dataset = TabularMultiViewDataset(
                X_scaled[train_idx],
                dataset.target[train_idx],  # type: ignore
                transform=base_transform,
                aug_transform=base_transform
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx],
                dataset.target[test_idx],  # type: ignore
                transform=base_transform,
                aug_transform=base_transform
            )
        else:
            self.train_dataset = TabularMultiViewDataset(
                X_scaled[train_idx],
                dataset.target[train_idx],  # type: ignore
                transform=base_transform,
                aug_transform=aug_transform
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx],
                dataset.target[test_idx],  # type: ignore
                transform=base_transform,
                aug_transform=base_transform
            )

        self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        self.ground_truth_labels = dataset.target  # type: ignore
        self.input_dim = dataset.data.shape[1]  # Number of features  # type: ignore

    def train_dataloader(self) -> DataLoader:
        if self.hparams.subset == 'all':
            dataset = self.full_dataset
        else:
            dataset = self.train_dataset
        return DataLoader(
            dataset,  # type: ignore
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,  # type: ignore
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def get_class_names(self) -> Dict[int, str]:
        return {
            0: "Setosa",
            1: "Versicolour", 
            2: "Virginica"
        }