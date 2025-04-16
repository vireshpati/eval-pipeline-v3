"""Base dataset interface for the benchmarking pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from torch.utils.data import DataLoader


class BenchmarkDataset(ABC):
    """Base class for all datasets in the benchmarking pipeline."""
    
    @abstractmethod
    def get_train_dataloader(self, config: Dict[str, Any]) -> DataLoader:
        """Returns a DataLoader for the training dataset.
        
        Args:
            config: Configuration dictionary with dataset parameters.
            
        Returns:
            DataLoader for the training dataset.
        """
        pass
    
    @abstractmethod
    def get_valid_dataloader(self, config: Dict[str, Any]) -> DataLoader:
        """Returns a DataLoader for the validation dataset.
        
        Args:
            config: Configuration dictionary with dataset parameters.
            
        Returns:
            DataLoader for the validation dataset.
        """
        pass
    
    @abstractmethod
    def get_test_dataloader(self, config: Dict[str, Any]) -> DataLoader:
        """Returns a DataLoader for the test dataset.
        
        Args:
            config: Configuration dictionary with dataset parameters.
            
        Returns:
            DataLoader for the test dataset.
        """
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Callable]:
        """Returns a dictionary of metric functions for evaluation.
        
        Returns:
            Dictionary mapping metric names to metric functions.
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Returns the vocabulary size for the dataset.
        
        Returns:
            Vocabulary size.
        """
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Returns the number of classes for classification tasks.
        
        Returns:
            Number of classes.
        """
        pass
