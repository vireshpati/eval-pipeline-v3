"""Image Modeling dataset implementation for the benchmarking pipeline."""

import os
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
import random

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

from datasets.base_dataset import BenchmarkDataset


class DummyImageNetDataset(Dataset):
    """Dummy ImageNet dataset for testing and scaffolding."""
    
    def __init__(self, 
                 split: str = 'train',
                 image_size: int = 224,
                 num_classes: int = 1000):
        """Initialize the dummy dataset.
        
        Args:
            split: Data split ('train', 'valid', or 'test').
            image_size: Size of images.
            num_classes: Number of classes.
        """
        super().__init__()
        
        self.split = split
        self.image_size = image_size
        self.num_classes = num_classes
        
        # Create dummy data
        # For testing, create at least 10 samples to avoid empty dataset errors
        self.num_samples = 20 if split == 'train' else 10
        
        # Create class names
        self.classes = [f"class_{i}" for i in range(num_classes)]
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image, label).
        """
        # Generate a random label
        label = idx % self.num_classes
        
        # Generate random image
        # Shape: [channels, height, width]
        image = torch.randn(3, self.image_size, self.image_size)
        
        return image, label


class ImageNet1kDataset(Dataset):
    """ImageNet-1k dataset."""
    
    def __init__(self, 
                 root: str, 
                 split: str = 'train',
                 image_size: int = 224,
                 transform: Optional[Callable] = None):
        """Initialize the dataset.
        
        Args:
            root: Root directory of the dataset.
            split: Data split ('train', 'valid', or 'test').
            image_size: Size of images.
            transform: Optional transform to apply to images.
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.image_size = image_size
        
        # Map split names to ImageFolder structure
        split_map = {
            'train': 'train',
            'valid': 'val',
            'test': 'val'  # Use validation set for testing as well
        }
        
        # Default transforms if none provided
        if transform is None:
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(int(image_size * 1.14)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Check if dataset exists
        split_dir = os.path.join(root, split_map[split])
        if not os.path.exists(split_dir) or not os.listdir(split_dir):
            print(f"Warning: Dataset directory not found or empty at {split_dir}")
            print(f"Creating dummy data for scaffolding purposes.")
            self._create_dummy_classes(split_dir)
            self.dummy_dataset = DummyImageNetDataset(
                split=split,
                image_size=image_size
            )
            return
        
        # Load dataset
        try:
            self.dataset = datasets.ImageFolder(
                root=split_dir,
                transform=self.transform
            )
            self.classes = self.dataset.classes
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(f"Creating dummy data for scaffolding purposes.")
            self._create_dummy_classes(split_dir)
            self.dummy_dataset = DummyImageNetDataset(
                split=split,
                image_size=image_size
            )
    
    def _create_dummy_classes(self, split_dir):
        """Create dummy class directories for ImageFolder compatibility."""
        # Create at least 10 class directories
        os.makedirs(split_dir, exist_ok=True)
        for i in range(10):
            class_dir = os.path.join(split_dir, f"class_{i}")
            os.makedirs(class_dir, exist_ok=True)
            
            # Create a dummy image file in each class directory
            # This is just to make ImageFolder happy, we'll still use the dummy dataset
            dummy_file = os.path.join(class_dir, "dummy.txt")
            if not os.path.exists(dummy_file):
                with open(dummy_file, 'w') as f:
                    f.write("Dummy file for ImageFolder compatibility")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        if hasattr(self, 'dummy_dataset'):
            return len(self.dummy_dataset)
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (image, label).
        """
        # Return dummy data if dataset doesn't exist
        if hasattr(self, 'dummy_dataset'):
            return self.dummy_dataset[idx]
        
        # Get sample from ImageFolder
        try:
            return self.dataset[idx]
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return dummy image and label on error
            return torch.randn(3, self.image_size, self.image_size), 0


class ImageModelingDataset(BenchmarkDataset):
    """Image modeling dataset implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset with the given configuration.
        
        Args:
            config: Configuration dictionary with dataset parameters.
        """
        super().__init__()
        
        self.config = config
        self.data_dir = config.get('data_dir', 'data/imagenet1k_subset/')
        self.image_size = config.get('image_size', 224)
        self.num_classes = config.get('num_classes', 1000)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create dataset instances
        self._create_datasets()
    
    def _create_datasets(self):
        """Create dataset instances for train, valid, and test splits."""
        self.train_dataset = ImageNet1kDataset(
            root=self.data_dir,
            split='train',
            image_size=self.image_size
        )
        
        self.valid_dataset = ImageNet1kDataset(
            root=self.data_dir,
            split='valid',
            image_size=self.image_size
        )
        
        self.test_dataset = ImageNet1kDataset(
            root=self.data_dir,
            split='test',
            image_size=self.image_size
        )
    
    def get_train_dataloader(self, config: Dict[str, Any]) -> DataLoader:
        """Returns a DataLoader for the training dataset.
        
        Args:
            config: Configuration dictionary with dataset parameters.
            
        Returns:
            DataLoader for the training dataset.
        """
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)
        
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def get_valid_dataloader(self, config: Dict[str, Any]) -> DataLoader:
        """Returns a DataLoader for the validation dataset.
        
        Args:
            config: Configuration dictionary with dataset parameters.
            
        Returns:
            DataLoader for the validation dataset.
        """
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)
        
        return DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def get_test_dataloader(self, config: Dict[str, Any]) -> DataLoader:
        """Returns a DataLoader for the test dataset.
        
        Args:
            config: Configuration dictionary with dataset parameters.
            
        Returns:
            DataLoader for the test dataset.
        """
        batch_size = config.get('batch_size', 32)
        num_workers = config.get('num_workers', 4)
        
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to handle different input formats.
        
        Args:
            batch: List of samples from the dataset.
            
        Returns:
            Dictionary containing batched tensors.
        """
        images = []
        labels = []
        
        for item in batch:
            if isinstance(item, tuple) and len(item) == 2:
                image, label = item
                images.append(image)
                labels.append(label)
            else:
                # Handle unexpected input format
                print(f"Warning: Unexpected item format in batch: {type(item)}")
                # Create dummy data
                images.append(torch.randn(3, self.image_size, self.image_size))
                labels.append(0)
        
        # Stack tensors
        images = torch.stack(images)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': images,
            'labels': labels
        }
    
    def get_metrics(self) -> Dict[str, Callable]:
        """Returns a dictionary of metric functions for evaluation.
        
        Returns:
            Dictionary mapping metric names to metric functions.
        """
        def accuracy(outputs, batch):
            logits = outputs['logits']
            if isinstance(batch, dict):
                labels = batch['labels']
            else:
                labels = batch[1]
            
            # Top-1 accuracy
            preds = torch.argmax(logits, dim=1)
            top1_acc = (preds == labels).float().mean()
            
            # Top-5 accuracy
            _, top5_preds = torch.topk(logits, k=5, dim=1)
            top5_correct = torch.zeros_like(labels, dtype=torch.float32)
            for i in range(top5_preds.size(0)):
                top5_correct[i] = (top5_preds[i] == labels[i]).any().float()
            top5_acc = top5_correct.mean()
            
            return {
                'top1': top1_acc,
                'top5': top5_acc
            }
        
        return {
            'accuracy': accuracy
        }
    
    def get_vocab_size(self) -> int:
        """Returns the vocabulary size for the dataset.
        
        Returns:
            Vocabulary size.
        """
        # Not applicable for image classification
        return 0
    
    def get_num_classes(self) -> int:
        """Returns the number of classes for classification tasks.
        
        Returns:
            Number of classes.
        """
        return self.num_classes
