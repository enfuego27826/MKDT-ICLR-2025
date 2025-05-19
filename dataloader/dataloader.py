import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from augmentations import get_ssl_augmentations

def get_ssl_dataloaders(dataset_name,batch_size,aug_type='barlow_twins',data_root='./data',train=True):
    """Load dataset with SSL augmentations applied (returns x1, x2 pairs)"""

    image_size = 32 if 'cifar' in dataset_name else 64
    transform = get_ssl_augmentations(aug_type=aug_type,image_size=image_size)

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root = data_root,train=train,download=True,transform=transform)
    
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_root,train=train,download=False,transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}, choose one of cifar10 or cifar100")

    return DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=2,drop_last=True)