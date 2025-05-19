import torch
from torchvision import transforms
import random

class SSLTransform:
    """Return two augmented views of an image, as required by SSL methods."""

    def __init__(self,aug_type='barlow_twins',image_size=32):
        """Initialize augmentation pipeline."""

        self.aug_type = aug_type

        if aug_type == 'barlow_twins':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size,scale=(0.2,1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4,0.4,0.4,0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor()
            ])
        
        elif aug_type == 'simclr':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size,scale=(0.08,1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.8,0.8,0.8,0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor()
            ])

        elif aug_type == 'none':
            #Augmentations for linear probing data or baseline
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        
        else:
            raise ValueError(f"Unsupported Augmentation Type: {aug_type}")

    
    def __call__(self,x,aug_type,image_size):
        """
        Apply the transform twice.
        """

        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1,x2

def get_ssl_augmentations(aug_type='barlow_twins',image_size=32):
    """Return SSLTransform"""
    return SSLTransform(aug_type=aug_type,image_size=image_size)
