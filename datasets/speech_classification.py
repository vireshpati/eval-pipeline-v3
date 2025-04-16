"""Speech Classification dataset implementation for the benchmarking pipeline."""

import os
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader
import torchaudio

from datasets.base_dataset import BenchmarkDataset


class DummySC10Dataset(Dataset):
    """Dummy SC10 dataset for testing and scaffolding."""
    
    def __init__(self, 
                 split: str = 'train',
                 n_mels: int = 80,
                 sample_rate: int = 16000,
                 duration: float = 1.0):
        """Initialize the dummy dataset.
        
        Args:
            split: Data split ('train', 'valid', or 'test').
            n_mels: Number of mel filterbanks.
            sample_rate: Audio sample rate.
            duration: Duration of audio clips in seconds.
        """
        super().__init__()
        
        self.split = split
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.duration = duration
        
        # Create dummy data
        # For testing, create at least 10 samples to avoid empty dataset errors
        self.num_samples = 20 if split == 'train' else 10
        self.num_classes = 10  # SC10 has 10 classes
        
        # Class names for SC10
        self.classes = [
            'yes', 'no', 'up', 'down', 'left', 
            'right', 'on', 'off', 'stop', 'go'
        ]
    
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
            Tuple of (audio_features, label).
        """
        # Generate a random label
        label = idx % self.num_classes
        
        # Generate random mel spectrogram features
        # Shape: [n_mels, time]
        time_steps = int(self.duration * self.sample_rate / 160)  # Assuming hop_length=160
        features = torch.randn(self.n_mels, time_steps)
        
        return features, label


class SC10Dataset(Dataset):
    """Google Speech Commands v1 dataset (SC10 subset)."""
    
    def __init__(self, 
                 root: str, 
                 split: str = 'train',
                 n_mels: int = 80,
                 sample_rate: int = 16000,
                 transform: Optional[Callable] = None):
        """Initialize the dataset.
        
        Args:
            root: Root directory of the dataset.
            split: Data split ('train', 'valid', or 'test').
            n_mels: Number of mel filterbanks.
            sample_rate: Audio sample rate.
            transform: Optional transform to apply to audio.
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.transform = transform
        
        # SC10 classes (subset of Speech Commands)
        self.classes = [
            'yes', 'no', 'up', 'down', 'left', 
            'right', 'on', 'off', 'stop', 'go'
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load dataset
        self._load_dataset()
        
        # Check if dataset is empty or doesn't exist
        if not self.samples:
            print(f"Warning: No audio files found in {root} or directory not found.")
            print(f"Creating dummy data for scaffolding purposes.")
            self.dummy_dataset = DummySC10Dataset(
                split=split,
                n_mels=n_mels,
                sample_rate=sample_rate
            )
            return
        
        # Set up audio processing
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=400,
            hop_length=160,
            f_min=0,
            f_max=8000,
            power=2.0,
        )
        
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
    
    def _load_dataset(self):
        """Load the dataset from disk."""
        self.samples = []
        
        # Check if root directory exists
        if not os.path.exists(self.root):
            return
        
        # Check each class directory
        for class_name in self.classes:
            class_dir = os.path.join(self.root, class_name)
            if not os.path.exists(class_dir):
                continue
            
            # Get all audio files
            files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
            
            # If no audio files found, create a dummy file for testing
            if not files and (self.split == 'train' or self.split == 'test'):
                # Create a dummy file for testing purposes
                dummy_path = os.path.join(class_dir, f"dummy_{class_name}.wav")
                self.samples.append((
                    dummy_path,
                    self.class_to_idx[class_name]
                ))
                continue
                
            # Add real samples if they exist
            for filename in files:
                self.samples.append((
                    os.path.join(class_dir, filename),
                    self.class_to_idx[class_name]
                ))
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        if hasattr(self, 'dummy_dataset'):
            return len(self.dummy_dataset)
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Tuple of (audio_features, label).
        """
        # Return dummy data if dataset doesn't exist
        if hasattr(self, 'dummy_dataset'):
            return self.dummy_dataset[idx]
        
        # Get sample
        audio_path, label = self.samples[idx]
        
        # Check if file exists, if not return dummy features
        if not os.path.exists(audio_path):
            # Return dummy features
            time_steps = int(1.0 * self.sample_rate / 160)  # Assuming hop_length=160
            features = torch.randn(self.n_mels, time_steps)
            return features, label
        
        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.sample_rate
                )
                waveform = resampler(waveform)
            
            # Apply transform if provided
            if self.transform is not None:
                waveform = self.transform(waveform)
            
            # Convert to mel spectrogram
            mel_spectrogram = self.mel_transform(waveform)
            
            # Convert to dB
            log_mel_spectrogram = self.amplitude_to_db(mel_spectrogram)
            
            # Remove batch dimension
            features = log_mel_spectrogram.squeeze(0)
            
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return dummy features on error
            time_steps = int(1.0 * self.sample_rate / 160)  # Assuming hop_length=160
            features = torch.randn(self.n_mels, time_steps)
        
        return features, label


class SpeechClassificationDataset(BenchmarkDataset):
    """Speech classification dataset implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset with the given configuration.
        
        Args:
            config: Configuration dictionary with dataset parameters.
        """
        super().__init__()
        
        self.config = config
        self.data_dir = config.get('data_dir', 'data/sc10/')
        self.n_mels = config.get('n_mels', 80)
        self.sample_rate = config.get('sample_rate', 16000)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create dataset instances
        self._create_datasets()
    
    def _create_datasets(self):
        """Create dataset instances for train, valid, and test splits."""
        self.train_dataset = SC10Dataset(
            root=self.data_dir,
            split='train',
            n_mels=self.n_mels,
            sample_rate=self.sample_rate
        )
        
        self.valid_dataset = SC10Dataset(
            root=self.data_dir,
            split='valid',
            n_mels=self.n_mels,
            sample_rate=self.sample_rate
        )
        
        self.test_dataset = SC10Dataset(
            root=self.data_dir,
            split='test',
            n_mels=self.n_mels,
            sample_rate=self.sample_rate
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
    
    def _collate_fn(self, batch: List[Tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching samples.
        
        Args:
            batch: List of (audio_features, label) tuples.
            
        Returns:
            Dictionary containing:
                - 'input_ids': Batched audio features.
                - 'labels': Batched labels.
        """
        # Extract features and labels
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # Get max sequence length
        max_length = max(feat.size(1) for feat in features)
        
        # Pad features to max length
        padded_features = []
        for feat in features:
            # [n_mels, time] -> [n_mels, max_length]
            padded_feat = torch.zeros(self.n_mels, max_length, dtype=feat.dtype)
            padded_feat[:, :feat.size(1)] = feat
            padded_features.append(padded_feat)
        
        # Stack tensors
        # [batch_size, n_mels, max_length] -> [batch_size, max_length, n_mels]
        features_tensor = torch.stack(padded_features).transpose(1, 2)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        return {
            'input_ids': features_tensor,
            'labels': labels_tensor
        }
    
    def get_metrics(self) -> Dict[str, Callable]:
        """Returns a dictionary of metric functions for evaluation.
        
        Returns:
            Dictionary mapping metric names to metric functions.
        """
        def accuracy(outputs, batch):
            logits = outputs['logits']
            labels = batch['labels'] if isinstance(batch, dict) else batch[1]
            preds = torch.argmax(logits, dim=1)
            return (preds == labels).float().mean()
        
        return {
            'accuracy': accuracy
        }
    
    def get_vocab_size(self) -> int:
        """Returns the vocabulary size for the dataset.
        
        Returns:
            Vocabulary size.
        """
        # Not applicable for speech classification
        return 0
    
    def get_num_classes(self) -> int:
        """Returns the number of classes for classification tasks.
        
        Returns:
            Number of classes.
        """
        return len(self.train_dataset.classes)
