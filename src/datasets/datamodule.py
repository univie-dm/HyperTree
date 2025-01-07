import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import torch
from typing import Literal, Optional, Union
import torch
import src.datasets.optdigits as optdigits


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_transform, aug_transform):
        self.dataset = dataset
        self.base_transform = base_transform
        self.aug_transform = aug_transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.aug_transform(img), self.base_transform(img), label

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
            "optdigits": optdigits.Optdigits

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
