import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch
from typing import Literal, Optional, Union
import torch
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, TensorDataset


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_transform, aug_transform):
        self.dataset = dataset
        self.base_transform = base_transform
        self.aug_transform = aug_transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Apply transforms
        aug_img = self.aug_transform(img)
        base_img = self.base_transform(img)
        
        # Flatten the images if they're not already flat
        if len(aug_img.shape) > 1:  # If image is 2D or 3D
            aug_img = aug_img.view(-1)
        if len(base_img.shape) > 1:
            base_img = base_img.view(-1)
            
        return aug_img, base_img, label

    def __len__(self):
        return len(self.dataset)

class ColorDataModule(pl.LightningDataModule):
    @staticmethod
    def get_normalize_fn(data: torch.Tensor):
        channel_means = data.mean([0, 2, 3])
        channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(channel_means, channel_stds)

    def __init__(
        self,
        dataset_name: Literal["cifar10", "cifar100", "cifar100_20", "gtsrb"] = "cifar10",
        data_dir: str = './data',
        batch_size: int = 256,
        resize: Optional[int] = None,
        subset: str = 'train',
        num_workers: int = 1
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.subset = subset
        self.num_workers = num_workers
        self.dataset_map = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
            "gtsrb": datasets.GTSRB,
        }
        self.train_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        self.ground_truth_labels = None
        
    def setup(self, stage: str = None):
        dataset_class = self.dataset_map[self.dataset_name]
        
        # Special handling for CIFAR100_20
        if self.dataset_name == "cifar100_20":
            dataset_class = datasets.CIFAR100
            self.coarse_labels = True
        else:
            self.coarse_labels = False

        # train parameter is not available in GTSRB, its replaced by split
        base_transforms = [transforms.ToTensor()]
        if self.dataset_name == "gtsrb":
            # Force resize for GTSRB to handle inconsistent image sizes
            base_transforms.insert(0, transforms.Resize((32, 32)))
            temp_dataset = dataset_class(self.data_dir, split='train', 
                                      transform=transforms.Compose(base_transforms), 
                                      download=True)
        else:
            temp_dataset = dataset_class(self.data_dir, train=True,
                                       transform=transforms.Compose(base_transforms))
        temp_loader = DataLoader(temp_dataset, batch_size=1000, num_workers=self.num_workers)
        normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])
        
        transform_list = []
        if self.resize:
            transform_list.append(transforms.Resize((self.resize, self.resize)))
        transform_list.extend([transforms.ToTensor(), normalize_fn])
        base_transform = transforms.Compose(transform_list)
        
        aug_transform = transforms.Compose([
            *transform_list[:-2],
            transforms.RandomResizedCrop((self.resize, self.resize) if self.resize else (32, 32)),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], 
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomSolarize(threshold=0.5, p=0.2),
            transforms.ToTensor(),
            normalize_fn
        ])

        # here we go again
        if self.dataset_name == "gtsrb":
            base_train = dataset_class(self.data_dir, split='train', transform=None, download=True)
            base_test = dataset_class(self.data_dir, split='test', transform=None, download=True)
        else:
            base_train = dataset_class(self.data_dir, train=True, transform=None)
            base_test = dataset_class(self.data_dir, train=False, transform=None)
        
        # Handle CIFAR100 coarse labels (20 classes)
        if self.coarse_labels:
            base_train.targets = [self._get_coarse_label(t) for t in base_train.targets]
            base_test.targets = [self._get_coarse_label(t) for t in base_test.targets]
        
        if self.subset == 'all':
            self.train_dataset = MultiViewDataset(base_train, base_transform, base_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
        else:
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
        self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        if "targets" in base_test.__dict__:
            self.ground_truth_labels = base_test.targets
        else:
            labels = []
            for _, _, label in self.test_dataset:
                labels.append(label)
            self.ground_truth_labels = torch.tensor(labels)

    def train_dataloader(self):
        if self.subset == 'all':
            dataset = self.full_dataset
        else:
            dataset = self.train_dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    @staticmethod
    def _get_coarse_label(fine_label):
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

class GrayScaleDataModule(pl.LightningDataModule):
    @staticmethod
    def get_normalize_fn(data: torch.Tensor):
        channel_means = data.mean([0, 2, 3])
        channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(channel_means, channel_stds)

    def __init__(
        self,
        dataset_name: Literal["mnist", "fmnist", "kmnist", "usps", "optdigits"] = "mnist",
        data_dir: str = './data',
        batch_size: int = 256,
        resize: int = 28,
        subset: str = 'train',
        num_workers: int = 1
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.subset = subset
        self.num_workers = num_workers
        self.train_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        self.ground_truth_labels = None
        self.dataset_map = {
            "mnist": datasets.MNIST,
            "fmnist": datasets.FashionMNIST,
            "kmnist": datasets.KMNIST,
            "usps": datasets.USPS,

        }

    def setup(self, stage: str = None):
        dataset_class = self.dataset_map[self.dataset_name]
        
        dataset_class(self.data_dir, train=False, transform=transforms.ToTensor(), download=True)
        temp_dataset = dataset_class(self.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        temp_loader = DataLoader(temp_dataset, batch_size=1000, num_workers=self.num_workers)
        normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])
        
        transform_list = []
        if self.resize:
            transform_list.append(transforms.Resize((self.resize, self.resize)))
        transform_list.extend([transforms.ToTensor(), normalize_fn])
        base_transform = transforms.Compose(transform_list)
        
        aug_transform = transforms.Compose([
            *transform_list[:-2],
            transforms.RandomAffine(degrees=16, translate=(0.1, 0.1), shear=8),
            transforms.ToTensor(),
            normalize_fn
        ])

        base_test = dataset_class(self.data_dir, train=False, transform=None)
        base_train = dataset_class(self.data_dir, train=True, transform=None)
        
        if self.subset == 'all':
            self.train_dataset = MultiViewDataset(base_train, base_transform, base_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
        else:
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
        self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        self.ground_truth_labels = base_test.targets

    def train_dataloader(self):
        if self.subset == 'all':
            dataset = self.full_dataset
        else:
            dataset = self.train_dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                         num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers)








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
    def __init__(self, noise_std=0.0):
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
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.subset = subset
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.aug_noise_std = aug_noise_std
        
        self.train_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        self.ground_truth_labels = None
        self.input_dim = None
        
        # Map dataset names to loading functions
        self.dataset_map = {
            "iris": load_iris
        }

    def prepare_data(self):
        # Download datasets if needed (not needed for iris)
        pass

    def setup(self, stage: str = None):
        # Load dataset
        dataset_fn = self.dataset_map[self.dataset_name]
        dataset = dataset_fn()
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dataset.data)
        
        # Split into train/test
        num_samples = len(dataset.data)
        num_train = int(num_samples * self.train_ratio)
        indices = np.random.permutation(num_samples)
        train_idx, test_idx = indices[:num_train], indices[num_train:]
        
        # Create transforms
        base_transform = TabularTransform(noise_std=0.0)
        aug_transform = TabularTransform(noise_std=self.aug_noise_std)
        
        # Create train/test datasets
        if self.subset == 'all':
            self.train_dataset = TabularMultiViewDataset(
                X_scaled[train_idx], 
                dataset.target[train_idx],
                transform=base_transform,
                aug_transform=base_transform
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx], 
                dataset.target[test_idx],
                transform=base_transform,
                aug_transform=base_transform
            )
        else:
            self.train_dataset = TabularMultiViewDataset(
                X_scaled[train_idx], 
                dataset.target[train_idx],
                transform=base_transform,
                aug_transform=aug_transform
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx], 
                dataset.target[test_idx],
                transform=base_transform,
                aug_transform=base_transform
            )
            
        self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        self.ground_truth_labels = dataset.target
        self.input_dim = dataset.data.shape[1]  # Number of features

    def train_dataloader(self):
        if self.subset == 'all':
            dataset = self.full_dataset
        else:
            dataset = self.train_dataset
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )