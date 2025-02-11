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
                                       transform=transforms.Compose(base_transforms,), download=True)
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
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
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
        elif self.subset == 'smallest':
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                         num_workers=self.num_workers)


    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                         num_workers=self.num_workers)

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
    

    def get_class_names(self):
        # For CIFAR10:
        if self.dataset_name == "cifar10":
            fine_labels = datasets.CIFAR10(self.data_dir, train=False).classes
            return {i: (fine_labels[i], None) for i in range(len(fine_labels))}
        # For CIFAR100:
        elif self.dataset_name == "cifar100":
            fine_labels = datasets.CIFAR100(self.data_dir, train=False).classes
            
            # CIFAR100 superclass mapping
            coarse_labels = [
                'aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals',
                'trees', 'vehicles 1', 'vehicles 2'
            ]
            
            # Mapping of fine classes to their superclasses (index corresponds to coarse_labels)
            fine_to_coarse = {
                'beaver': 0, 'dolphin': 0, 'otter': 0, 'seal': 0, 'whale': 0,
                'aquarium_fish': 1, 'flatfish': 1, 'ray': 1, 'shark': 1, 'trout': 1,
                'orchid': 2, 'poppy': 2, 'rose': 2, 'sunflower': 2, 'tulip': 2,
                'bottle': 3, 'bowl': 3, 'can': 3, 'cup': 3, 'plate': 3,
                'apple': 4, 'mushroom': 4, 'orange': 4, 'pear': 4, 'sweet_pepper': 4,
                'clock': 5, 'keyboard': 5, 'lamp': 5, 'telephone': 5, 'television': 5,
                'bed': 6, 'chair': 6, 'couch': 6, 'table': 6, 'wardrobe': 6,
                'bee': 7, 'beetle': 7, 'butterfly': 7, 'caterpillar': 7, 'cockroach': 7,
                'bear': 8, 'leopard': 8, 'lion': 8, 'tiger': 8, 'wolf': 8,
                'bridge': 9, 'castle': 9, 'house': 9, 'road': 9, 'skyscraper': 9,
                'cloud': 10, 'forest': 10, 'mountain': 10, 'plain': 10, 'sea': 10,
                'camel': 11, 'cattle': 11, 'chimpanzee': 11, 'elephant': 11, 'kangaroo': 11,
                'fox': 12, 'porcupine': 12, 'possum': 12, 'raccoon': 12, 'skunk': 12,
                'crab': 13, 'lobster': 13, 'snail': 13, 'spider': 13, 'worm': 13,
                'baby': 14, 'boy': 14, 'girl': 14, 'man': 14, 'woman': 14,
                'crocodile': 15, 'dinosaur': 15, 'lizard': 15, 'snake': 15, 'turtle': 15,
                'hamster': 16, 'mouse': 16, 'rabbit': 16, 'shrew': 16, 'squirrel': 16,
                'maple_tree': 17, 'oak_tree': 17, 'palm_tree': 17, 'pine_tree': 17, 'willow_tree': 17,
                'bicycle': 18, 'bus': 18, 'motorcycle': 18, 'pickup_truck': 18, 'train': 18,
                'lawn_mower': 19, 'rocket': 19, 'streetcar': 19, 'tank': 19, 'tractor': 19
            }
            
            return {i: (fine_labels[i], coarse_labels[fine_to_coarse[fine_labels[i]]]) 
                    for i in range(len(fine_labels))}

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
    
    def get_class_names(self):
        # For MNIST:
        if self.name == "mnist":
            return {
                0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
                5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
            }
        # For FashionMNIST:
        elif self.name == "fmnist":
            return {
                0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
            }
        else:
            None








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
    
    def get_class_names(self):
        return {
            0: "Setosa",
            1: "Versicolour",
            2: "Virginica"
        }