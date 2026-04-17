"""
Data Loading and Preprocessing for Retinal Fundus Images

Handles dataset loading, train/val/test splitting,
and image augmentation for DR grading tasks.
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FundusDataset(Dataset):
    """Custom dataset for retinal fundus images.
    
    Args:
        image_dir: Path to image directory
        labels: Dictionary mapping filenames to DR grades (0-4)
        transform: torchvision transforms to apply
    """
    
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.image_files = list(labels.keys())
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        label = self.labels[filename]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(image_size=224, is_training=True):
    """Get image transforms for training or evaluation.
    
    Args:
        image_size: Target image size (default: 224)
        is_training: Whether to apply augmentation
    
    Returns:
        torchvision.transforms.Compose
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
