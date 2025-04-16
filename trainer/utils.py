"""Utility functions for the benchmarking pipeline."""

import os
import random
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get device for PyTorch.
    
    Returns:
        PyTorch device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_padding_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create padding mask from sequence lengths.
    
    Args:
        lengths: Tensor of sequence lengths.
        max_len: Maximum sequence length.
        
    Returns:
        Padding mask of shape [batch_size, max_len].
    """
    batch_size = lengths.size(0)
    if max_len is None:
        max_len = lengths.max().item()
    
    # Create position indices
    pos = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    
    # Create mask
    mask = pos < lengths.unsqueeze(1)
    
    return mask


def compute_accuracy(outputs: Dict[str, torch.Tensor], batch: Any) -> torch.Tensor:
    """Compute accuracy for classification task.
    
    Args:
        outputs: Model outputs from forward pass.
        batch: Input batch.
        
    Returns:
        Accuracy tensor.
    """
    # Get logits and labels
    logits = outputs['logits']
    
    # Get labels from batch
    if isinstance(batch, dict):
        labels = batch.get('labels', None)
    elif isinstance(batch, tuple) and len(batch) > 1:
        labels = batch[1]
    else:
        return torch.tensor(0.0, device=logits.device)
    
    # Compute predictions
    preds = logits.argmax(dim=-1)
    
    # Compute accuracy
    correct = (preds == labels).float()
    accuracy = correct.mean()
    
    return accuracy


def compute_bleu(outputs: Dict[str, torch.Tensor], batch: Any) -> torch.Tensor:
    """Compute BLEU score for translation task.
    
    This is a simplified version that returns a dummy score.
    In a real implementation, this would use a proper BLEU calculation.
    
    Args:
        outputs: Model outputs from forward pass.
        batch: Input batch.
        
    Returns:
        BLEU score tensor.
    """
    # In a real implementation, this would compute BLEU score
    # For now, return a dummy score
    return torch.tensor(0.5, device=outputs['logits'].device)


def compute_top_k_accuracy(outputs: Dict[str, torch.Tensor], 
                          batch: Any, 
                          k: int = 5) -> torch.Tensor:
    """Compute top-k accuracy for classification task.
    
    Args:
        outputs: Model outputs from forward pass.
        batch: Input batch.
        k: k value for top-k accuracy.
        
    Returns:
        Top-k accuracy tensor.
    """
    # Get logits and labels
    logits = outputs['logits']
    
    # Get labels from batch
    if isinstance(batch, dict):
        labels = batch.get('labels', None)
    elif isinstance(batch, tuple) and len(batch) > 1:
        labels = batch[1]
    else:
        return torch.tensor(0.0, device=logits.device)
    
    # Compute top-k predictions
    _, topk_preds = logits.topk(k, dim=-1)
    
    # Expand labels for comparison
    labels_expanded = labels.unsqueeze(-1).expand_as(topk_preds)
    
    # Compute accuracy
    correct = (topk_preds == labels_expanded).any(dim=-1).float()
    accuracy = correct.mean()
    
    return accuracy


def safe_mean(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Safely compute mean of tensor, handling empty tensors.
    
    Args:
        tensor: Input tensor.
        *args: Additional positional arguments for torch.mean.
        **kwargs: Additional keyword arguments for torch.mean.
        
    Returns:
        Mean tensor.
    """
    if tensor.numel() == 0:
        return torch.zeros((), device=tensor.device)
    return torch.mean(tensor, *args, **kwargs)


def safe_std(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Safely compute standard deviation of tensor, handling empty tensors.
    
    Args:
        tensor: Input tensor.
        *args: Additional positional arguments for torch.std.
        **kwargs: Additional keyword arguments for torch.std.
        
    Returns:
        Standard deviation tensor.
    """
    if tensor.numel() == 0 or tensor.size(0) <= 1:
        return torch.zeros((), device=tensor.device)
    return torch.std(tensor, *args, **kwargs)


def suppress_warnings() -> None:
    """Suppress PyTorch warnings, especially for std_mean operations."""
    import warnings
    import logging
    import os
    
    # Suppress PyTorch warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    os.environ["PYTHONWARNINGS"] = "ignore"
    
    # Suppress C++ extension warnings (for the ReduceOps.cpp warnings)
    logging.getLogger("torch._C").setLevel(logging.ERROR)
    os.environ["TORCH_CPP_LOG_LEVEL"] = "50"  # Critical level only
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"
    
    # Disable all logging below ERROR level
    logging.basicConfig(level=logging.ERROR)
    
    # Monkey patch torch.std to avoid warnings
    torch.std = safe_std
    torch.mean = safe_mean
    
    print("PyTorch warnings have been suppressed")
