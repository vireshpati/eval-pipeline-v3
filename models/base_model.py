"""Base model interface for the benchmarking pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BenchmarkModel(nn.Module, ABC):
    """Base class for all models in the benchmarking pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with the given configuration.
        
        Args:
            config: Configuration dictionary with model parameters.
        """
        super().__init__()
        self.config = config
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the model.
        
        Returns:
            Model name.
        """
        pass
    
    @abstractmethod
    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            batch: Input batch.
            
        Returns:
            Dictionary containing output tensors.
        """
        pass
    
    @abstractmethod
    def get_loss(self, outputs: Dict[str, torch.Tensor], batch: Any) -> torch.Tensor:
        """Compute loss for the given outputs and batch.
        
        Args:
            outputs: Model outputs from forward pass.
            batch: Input batch.
            
        Returns:
            Loss tensor.
        """
        pass
