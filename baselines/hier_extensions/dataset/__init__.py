from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .cifar100 import CIFAR100
from .tiny_imagenet import TinyImageNet
from .imagenet import ImageNet
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP,
    'CIFAR100': CIFAR100,
    'TinyImageNet': TinyImageNet,
    'ImageNet': ImageNet,
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
