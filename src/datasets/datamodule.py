import torch
import pytorch_lightning as pl
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from torchvision import transforms, datasets
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
from typing import Literal, Optional, Union


class SafeEqualize:
    """Custom transform that applies equalize to both PIL images and tensors."""
    def __call__(self, img):
        if hasattr(img, 'mode'):  # PIL Image
            return F.equalize(img)
        elif torch.is_tensor(img):
            # Convert tensor to uint8 format for equalization
            if img.dtype != torch.uint8:
                # Assume the tensor is in [0, 1] range if float, convert to [0, 255] uint8
                if img.dtype in [torch.float32, torch.float64]:
                    img_uint8 = (img * 255).clamp(0, 255).to(torch.uint8)
                else:
                    img_uint8 = img.to(torch.uint8)
            else:
                img_uint8 = img
            
            # Apply equalize and convert back to original dtype
            equalized = F.equalize(img_uint8)
            if img.dtype in [torch.float32, torch.float64]:
                return equalized.to(img.dtype) / 255.0
            else:
                return equalized.to(img.dtype)
        else:
            # If it's numpy array or other format, return as-is
            return img


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
    def get_normalize_fn(data: torch.Tensor) -> transforms.Normalize:
        channel_means = data.mean([0, 2, 3])
        channel_stds = data.std([0, 2, 3])
        return transforms.Normalize(channel_means, channel_stds)

    def __init__(
        self,
        dataset_name: Literal["cifar10", "cifar100", "cifar100_20", "gtsrb", "imagenet"] = "cifar10",
        data_dir: str = './data',
        batch_size: int = 256,
        resize: Optional[int] = None,
        subset: str = 'train',
        num_workers: int = 1,
        use_augmentations: bool = False,
        selected_classes: Optional[list] = None
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.subset = subset
        self.num_workers = num_workers
        self.use_augmentations = use_augmentations
        self.selected_classes = selected_classes if selected_classes is not None else []
        self.dataset_map = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
            "cifar100_20": datasets.CIFAR100,
            "gtsrb": datasets.GTSRB,
            "imagenet": datasets.ImageNet,
        }
        self.train_dataset = None
        self.test_dataset = None
        self.full_dataset = None
        self.ground_truth_labels = None

    def setup(self) -> None:
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
            base_transforms = [transforms.Resize((32, 32))] + base_transforms
            temp_dataset = dataset_class(self.data_dir, split='train', # type: ignore
                                         transform=transforms.Compose(base_transforms),
                                         download=True)
        elif self.dataset_name in ["cifar10", "cifar100", "cifar100_20"] and self.use_augmentations:
            base_transforms = [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4)
            ]
            temp_dataset = dataset_class(self.data_dir, train=True,
                                         transform=transforms.Compose(base_transforms,), download=True)
        elif self.dataset_name == "imagenet":
            # ImageNet specific preprocessing
            base_transforms = [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
            try:
                if dataset_class == datasets.ImageNet:
                    temp_dataset = dataset_class(self.data_dir, split='train', # type: ignore
                                           transform=transforms.Compose(base_transforms))
                else:
                    temp_dataset = dataset_class(self.data_dir, train=True,
                                           transform=transforms.Compose(base_transforms))
            except Exception as e:
                print(f"Error loading ImageNet dataset: {e}")
                print("Please make sure you have downloaded the ImageNet dataset to the specified data directory.")
                raise
        else:
            temp_dataset = dataset_class(self.data_dir, train=True,
                                         transform=transforms.Compose(base_transforms,), download=True)
            
        temp_loader = DataLoader(temp_dataset, batch_size=1000, num_workers=self.num_workers)
        normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])

        transform_list = []
        if self.resize:
            transform_list.append(transforms.Resize((self.resize, self.resize)))
        # Add SafeEqualize before ToTensor to work with PIL images
        transform_list.extend([transforms.ToTensor(), 
                               normalize_fn])
        base_transform = transforms.Compose(transform_list)

        # ImageNet specific augmentation
        if self.dataset_name == "imagenet":
            aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
                transforms.ToTensor(),
                normalize_fn
            ])
        elif self.dataset_name == "gtsrb":
            # GTSRB specific augmentation (already resized to 32x32)
            aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ToTensor(),
                normalize_fn
            ])
        else:
            # CIFAR and other datasets augmentation
            aug_transform = transforms.Compose([
                transforms.RandomResizedCrop(size=self.resize if self.resize else 32, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=3)
                ], p=0.5),
                transforms.ToTensor(),
                normalize_fn
            ])

        if self.dataset_name == "gtsrb":
            base_train = dataset_class(self.data_dir, split='train', transform=None, download=True) # type: ignore
            base_test = dataset_class(self.data_dir, split='test', transform=None, download=True) # type: ignore
        elif self.dataset_name == "imagenet":
            try:
                base_train = dataset_class(self.data_dir, split='train', transform=None) # type: ignore
                base_test = dataset_class(self.data_dir, split='val', transform=None) # type: ignore
            except Exception as e:
                print(f"Error loading ImageNet dataset: {e}")
                print("Please make sure you have downloaded the ImageNet dataset to the specified data directory.")
                raise
        else:
            base_train = dataset_class(self.data_dir, train=True, transform=None)
            base_test = dataset_class(self.data_dir, train=False, transform=None)

        # Handle CIFAR100 coarse labels (20 classes)
        if self.coarse_labels:
            base_train.targets = [self._get_coarse_label(t) for t in base_train.targets]
            base_test.targets = [self._get_coarse_label(t) for t in base_test.targets]

        if len(self.selected_classes) > 0:
            # Create masks for selected classes
            train_mask = torch.tensor([label in self.selected_classes for label in base_train.targets])
            test_mask = torch.tensor([label in self.selected_classes for label in base_test.targets])

            # Filter datasets
            base_train.data = base_train.data[train_mask]
            base_train.targets = [label for i, label in enumerate(base_train.targets) if train_mask[i]]
            base_test.data = base_test.data[test_mask]
            base_test.targets = [label for i, label in enumerate(base_test.targets) if test_mask[i]]

        if self.subset == 'all':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform if self.use_augmentations else base_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        elif self.subset == 'test':
            self.train_dataset = MultiViewDataset(base_test, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = self.test_dataset  # Don't concatenate for 'smallest'
        elif self.subset == 'train':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = self.train_dataset
        else:
            raise ValueError(f"Invalid subset: {self.subset}")

        if hasattr(base_test, 'targets'):
            self.ground_truth_labels = base_test.targets
        else:
            self.ground_truth_labels = [label for _, label in base_test]

    def train_dataloader(self) -> DataLoader:
        if self.subset == 'all':
            dataset = self.full_dataset
        elif self.subset == 'smallest':
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset

        if dataset is None:
            raise RuntimeError("Dataset not initialized. Please call setup() first.")

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Dataset not initialized. Please call setup() first.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

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

    def get_class_names(self) -> Union[None, dict]:
        # For CIFAR10:
        if self.dataset_name == "cifar10":
            fine_labels = datasets.CIFAR10(self.data_dir, train=False).classes
            return {i: (fine_labels[i], None) for i in range(len(fine_labels))}
        # For CIFAR100:
        elif self.dataset_name in ["cifar100", "cifar100_20"]:
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

            if self.dataset_name == "cifar100_20":
                # For CIFAR100_20, we only return the coarse labels
                return {i: (coarse_labels[i], None) for i in range(len(coarse_labels))}
            else:
                # For regular CIFAR100, return both fine and coarse labels
                return {i: (fine_labels[i], coarse_labels[fine_to_coarse[fine_labels[i]]])
                        for i in range(len(fine_labels))}
        # For GTSRB:
        elif self.dataset_name == "gtsrb":
            # GTSRB fine class names (43 classes)
            fine_labels = [
                "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", 
                "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", 
                "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", 
                "No passing", "No passing for vehicles over 3.5 metric tons", 
                "Right-of-way at the next intersection", "Priority road", "Yield", "Stop", 
                "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry", 
                "General caution", "Dangerous curve to the left", "Dangerous curve to the right", 
                "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right", 
                "Road work", "Traffic signals", "Pedestrians", "Children crossing", 
                "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing", 
                "End of all speed and passing limits", "Turn right ahead", "Turn left ahead", 
                "Ahead only", "Go straight or right", "Go straight or left", "Keep right", 
                "Keep left", "Roundabout mandatory", "End of no passing", 
                "End of no passing by vehicles over 3.5 metric tons"
            ]
            
            # GTSRB hierarchical categories
            coarse_labels = [
                "Prohibitory", "Mandatory", "Warning", "Speed Limits"
            ]
            
            # Mapping of fine classes to their hierarchical categories
            fine_to_coarse = {
                # Speed limit signs (0-8)
                0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3,
                # Prohibitory signs (9-17, 32, 41, 42)
                9: 0, 10: 0, 15: 0, 16: 0, 17: 0, 32: 0, 41: 0, 42: 0,
                # Priority/Right-of-way signs (11-14)
                11: 1, 12: 1, 13: 1, 14: 1,
                # Warning signs (18-31)
                18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2, 29: 2, 30: 2, 31: 2,
                # Mandatory signs (33-40)
                33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1, 40: 1
            }
            
            return {i: (fine_labels[i], coarse_labels[fine_to_coarse[i]]) 
                    for i in range(len(fine_labels))}
        # For ImageNet:
        elif self.dataset_name == "imagenet":
            # ImageNet class names are not directly available in torchvision
            # You would need to load them from a separate file or use a mapping
            return None
        else:
            return None

class GrayScaleDataModule(pl.LightningDataModule):
    @staticmethod
    def get_normalize_fn(data: torch.Tensor) -> transforms.Normalize:
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
        num_workers: int = 1,
        selected_classes: Optional[list] = None,
        use_augmentations: bool = False
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.resize = resize
        self.subset = subset
        self.num_workers = num_workers
        self.selected_classes = selected_classes
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
        self.use_augmentations = use_augmentations

    def setup(self) -> None:
        dataset_class = self.dataset_map[self.dataset_name]

        # First download the dataset
        dataset_class(self.data_dir, train=False, transform=transforms.ToTensor(), download=True)
        temp_dataset = dataset_class(self.data_dir, train=True, transform=transforms.ToTensor(), download=True)
        temp_loader = DataLoader(temp_dataset, batch_size=1000, num_workers=self.num_workers)
        normalize_fn = self.get_normalize_fn(next(iter(temp_loader))[0])

        transform_list = []
        if self.resize:
            transform_list.append(transforms.Resize((self.resize, self.resize)))
        transform_list.extend([transforms.ToTensor(), normalize_fn])
        base_transform = transforms.Compose(transform_list)
        if self.use_augmentations:
            # Define augmentations
            base_transform = transforms.Compose([
                *transform_list,
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(28, padding=4)
            ])
        else:
            base_transform = transforms.Compose(transform_list)

        aug_transform = transforms.Compose([
            *transform_list[:-2],
            transforms.RandomAffine(degrees=16, translate=(0.1, 0.1), shear=8),
            transforms.ToTensor(),
            normalize_fn
        ])

        # Load base datasets
        base_test = dataset_class(self.data_dir, train=False, transform=None)
        base_train = dataset_class(self.data_dir, train=True, transform=None)

        # Filter for selected classes if specified
        if self.selected_classes is not None:
            # Validate selected classes
            max_class = max(base_train.targets)
            invalid_classes = [c for c in self.selected_classes if c > max_class]
            if invalid_classes:
                raise ValueError(f"Invalid class labels for {self.dataset_name}: {invalid_classes}. "
                               f"Maximum class label is {max_class}.")

            # Create masks for selected classes
            train_mask = torch.tensor([label in self.selected_classes for label in base_train.targets])
            test_mask = torch.tensor([label in self.selected_classes for label in base_test.targets])

            # Filter datasets
            base_train.data = base_train.data[train_mask]
            base_train.targets = [label for i, label in enumerate(base_train.targets) if train_mask[i]]
            base_test.data = base_test.data[test_mask]
            base_test.targets = [label for i, label in enumerate(base_test.targets) if test_mask[i]]
        if self.subset == 'all':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform if self.use_augmentations else base_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        elif self.subset == 'smallest':
            self.train_dataset = MultiViewDataset(base_test, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = self.test_dataset  # Don't concatenate for 'smallest'
        elif self.subset == 'train':
            self.train_dataset = MultiViewDataset(base_train, base_transform, aug_transform)
            self.test_dataset = MultiViewDataset(base_test, base_transform, base_transform)
            self.full_dataset = self.train_dataset
        else:
            raise ValueError(f"Invalid subset: {self.subset}")

        self.ground_truth_labels = base_test.targets

    def train_dataloader(self) -> DataLoader:
        if self.subset == 'all':
            dataset = self.full_dataset
        elif self.subset == 'smallest':
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset
        if dataset is None:
            raise RuntimeError("Dataset not initialized. Please call setup() first.")
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Please call setup() first.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def get_class_names(self) -> Union[None, dict]:
        # For MNIST:
        if self.dataset_name == "mnist":
            return {
                0: "0-Zero", 1: "1-One", 2: "2-Two", 3: "3-Three", 4: "4-Four",
                5: "5-Five", 6: "6-Six", 7: "7-Seven", 8: "8-Eight", 9: "9-Nine"
            }
        # For FashionMNIST:
        elif self.dataset_name == "fmnist":
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

    def prepare_data(self) -> None:
        # Download datasets if needed (not needed for iris)
        pass

    def setup(self) -> None:
        # Load dataset
        dataset_fn = self.dataset_map[self.dataset_name]
        dataset = dataset_fn()

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(dataset.data) # type: ignore

        # Split into train/test
        num_samples = len(dataset.data) # type: ignore
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
                dataset.target[train_idx], # type: ignore
                transform=base_transform,
                aug_transform=base_transform
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx],
                dataset.target[test_idx], # type: ignore
                transform=base_transform,
                aug_transform=base_transform
            )
        else:
            self.train_dataset = TabularMultiViewDataset(
                X_scaled[train_idx],
                dataset.target[train_idx], # type: ignore
                transform=base_transform,
                aug_transform=aug_transform
            )
            self.test_dataset = TabularMultiViewDataset(
                X_scaled[test_idx],
                dataset.target[test_idx], # type: ignore
                transform=base_transform,
                aug_transform=base_transform
            )

        self.full_dataset = torch.utils.data.ConcatDataset([self.train_dataset, self.test_dataset])
        self.ground_truth_labels = dataset.target # type: ignore
        self.input_dim = dataset.data.shape[1]  # Number of features # type: ignore

    def train_dataloader(self) -> DataLoader:
        if self.subset == 'all':
            dataset = self.full_dataset
        else:
            dataset = self.train_dataset
        return DataLoader(
            dataset, # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset, # type: ignore
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def get_class_names(self) -> dict:
        return {
            0: "Setosa",
            1: "Versicolour",
            2: "Virginica"
        }

