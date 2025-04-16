"""Trainer implementation for the benchmarking pipeline."""

import os
import time
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Monkey patch torch.std to avoid warnings with empty tensors
original_std = torch.std
def safe_std(input, *args, **kwargs):
    if input.numel() == 0 or input.size(0) == 0:
        return torch.zeros((), device=input.device)
    return original_std(input, *args, **kwargs)
torch.std = safe_std


class Trainer:
    """Trainer for models in the benchmarking pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trainer with the given configuration.
        
        Args:
            config: Configuration dictionary with trainer parameters.
        """
        self.config = config
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 0.0)
        self.max_epochs = config.get('max_epochs', 10)
        self.max_steps = config.get('max_steps', None)
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Logging parameters
        self.log_every_n_steps = config.get('log_every_n_steps', 10)
        
        # Checkpointing parameters
        self.save_every_n_steps = config.get('save_every_n_steps', 100)
        self.save_dir = config.get('save_dir', 'checkpoints/')
        
        # Mixed precision parameters
        self.mixed_precision = config.get('mixed_precision', False)
        
        # Create save directory if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train(self, 
              model: nn.Module, 
              train_dataloader: DataLoader, 
              valid_dataloader: Optional[DataLoader] = None,
              metrics: Optional[Dict[str, Callable]] = None) -> Dict[str, float]:
        """Train the model.
        
        Args:
            model: Model to train.
            train_dataloader: DataLoader for training data.
            valid_dataloader: Optional DataLoader for validation data.
            metrics: Optional dictionary of metric functions.
            
        Returns:
            Dictionary of training metrics.
        """
        # Move model to device
        model.to(self.device)
        
        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_dataloader) * self.max_epochs
        )
        
        # Create scaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Training loop
        global_step = 0
        best_valid_loss = float('inf')
        
        for epoch in range(self.max_epochs):
            # Training epoch
            model.train()
            train_loss = 0.0
            train_metrics = {name: 0.0 for name in metrics.keys()} if metrics else {}
            
            start_time = time.time()
            
            for step, batch in enumerate(train_dataloader):
                # Move batch to device
                batch = self._move_batch_to_device(batch, self.device)
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast() if self.mixed_precision else torch.no_grad():
                    # Forward pass
                    outputs = model(batch)
                    
                    # Compute loss
                    loss = model.get_loss(outputs, batch)
                
                # Backward pass
                optimizer.zero_grad()
                
                if self.mixed_precision:
                    # Mixed precision backward pass
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping is removed to avoid warnings
                    
                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Standard backward pass
                    loss.backward()
                    
                    # Gradient clipping is removed to avoid warnings
                    
                    # Update parameters
                    optimizer.step()
                
                # Update learning rate
                scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                
                if metrics:
                    try:
                        for name, metric_fn in metrics.items():
                            # Handle different metric function signatures
                            if 'batch' in metric_fn.__code__.co_varnames:
                                metric_value = metric_fn(outputs, batch)
                            else:
                                metric_value = metric_fn(outputs)
                            
                            # Handle different return types
                            if isinstance(metric_value, dict):
                                for k, v in metric_value.items():
                                    train_metrics[f"{name}_{k}"] = train_metrics.get(f"{name}_{k}", 0.0) + v.item()
                            else:
                                train_metrics[name] = train_metrics.get(name, 0.0) + metric_value.item()
                    except Exception as e:
                        print(f"Error computing metrics: {e}")
                
                # Log progress
                if global_step % self.log_every_n_steps == 0:
                    # Calculate average metrics
                    avg_loss = train_loss / (step + 1)
                    avg_metrics = {name: value / (step + 1) for name, value in train_metrics.items()}
                    
                    # Log metrics
                    self._log_metrics(epoch, global_step, avg_loss, avg_metrics, 'train')
                
                # Save checkpoint
                if global_step % self.save_every_n_steps == 0:
                    self._save_checkpoint(model, optimizer, scheduler, epoch, global_step, train_loss)
                
                # Increment global step
                global_step += 1
                
                # Check if max steps reached
                if self.max_steps is not None and global_step >= self.max_steps:
                    break
            
            # Calculate average metrics for epoch
            avg_loss = train_loss / len(train_dataloader)
            avg_metrics = {name: value / len(train_dataloader) for name, value in train_metrics.items()}
            
            # Log epoch metrics
            epoch_time = time.time() - start_time
            self._log_epoch_metrics(epoch, global_step, avg_loss, avg_metrics, epoch_time, 'train')
            
            # Validation epoch
            if valid_dataloader is not None:
                valid_loss, valid_metrics = self.evaluate(model, valid_dataloader, metrics)
                
                # Log validation metrics
                self._log_epoch_metrics(epoch, global_step, valid_loss, valid_metrics, 0.0, 'valid')
                
                # Save best model
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    self._save_checkpoint(
                        model, optimizer, scheduler, epoch, global_step, valid_loss, 
                        filename='best_model.pt'
                    )
            
            # Check if max steps reached
            if self.max_steps is not None and global_step >= self.max_steps:
                break
        
        # Save final model
        self._save_checkpoint(
            model, optimizer, scheduler, epoch, global_step, train_loss,
            filename='final_model.pt'
        )
        
        # Return final metrics
        return {
            'loss': avg_loss,
            **avg_metrics
        }
    
    def evaluate(self, 
                model: nn.Module, 
                dataloader: DataLoader,
                metrics: Optional[Dict[str, Callable]] = None) -> Tuple[float, Dict[str, float]]:
        """Evaluate the model.
        
        Args:
            model: Model to evaluate.
            dataloader: DataLoader for evaluation data.
            metrics: Optional dictionary of metric functions.
            
        Returns:
            Tuple of (average loss, dictionary of evaluation metrics).
        """
        # Move model to device
        model.to(self.device)
        
        # Evaluation mode
        model.eval()
        
        # Initialize metrics
        eval_loss = 0.0
        eval_metrics = {name: 0.0 for name in metrics.keys()} if metrics else {}
        
        # Evaluation loop
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = self._move_batch_to_device(batch, self.device)
                
                # Forward pass
                outputs = model(batch)
                
                # Compute loss
                loss = model.get_loss(outputs, batch)
                
                # Update metrics
                eval_loss += loss.item()
                
                if metrics:
                    try:
                        for name, metric_fn in metrics.items():
                            # Handle different metric function signatures
                            if 'batch' in metric_fn.__code__.co_varnames:
                                metric_value = metric_fn(outputs, batch)
                            else:
                                metric_value = metric_fn(outputs)
                            
                            # Handle different return types
                            if isinstance(metric_value, dict):
                                for k, v in metric_value.items():
                                    eval_metrics[f"{name}_{k}"] = eval_metrics.get(f"{name}_{k}", 0.0) + v.item()
                            else:
                                eval_metrics[name] = eval_metrics.get(name, 0.0) + metric_value.item()
                    except Exception as e:
                        print(f"Error computing metrics: {e}")
        
        # Calculate average metrics
        avg_loss = eval_loss / len(dataloader)
        avg_metrics = {name: value / len(dataloader) for name, value in eval_metrics.items()}
        
        return avg_loss, avg_metrics
    
    def _move_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """Move batch to device.
        
        Args:
            batch: Batch to move.
            device: Device to move batch to.
            
        Returns:
            Batch on device.
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._move_batch_to_device(v, device) for v in batch]
        elif isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(v, device) for v in batch)
        else:
            return batch
    
    def _log_metrics(self, 
                    epoch: int, 
                    step: int, 
                    loss: float, 
                    metrics: Dict[str, float],
                    prefix: str = 'train') -> None:
        """Log metrics.
        
        Args:
            epoch: Current epoch.
            step: Current step.
            loss: Loss value.
            metrics: Dictionary of metric values.
            prefix: Prefix for metric names.
        """
        # Format metrics string
        metrics_str = f"loss: {loss:.4f}"
        for name, value in metrics.items():
            metrics_str += f", {name}: {value:.4f}"
        
        # Log metrics
        print(f"Epoch {epoch}, Step {step}, {prefix.capitalize()} {metrics_str}")
    
    def _log_epoch_metrics(self, 
                          epoch: int, 
                          step: int, 
                          loss: float, 
                          metrics: Dict[str, float],
                          time: float,
                          prefix: str = 'train') -> None:
        """Log epoch metrics.
        
        Args:
            epoch: Current epoch.
            step: Current step.
            loss: Loss value.
            metrics: Dictionary of metric values.
            time: Time taken for epoch.
            prefix: Prefix for metric names.
        """
        # Format metrics string
        metrics_str = f"loss: {loss:.4f}"
        for name, value in metrics.items():
            metrics_str += f", {name}: {value:.4f}"
        
        # Log metrics
        print(f"Epoch {epoch} completed, {prefix.capitalize()} {metrics_str}, Time: {time:.2f}s")
    
    def _save_checkpoint(self, 
                        model: nn.Module, 
                        optimizer: optim.Optimizer, 
                        scheduler: optim.lr_scheduler._LRScheduler, 
                        epoch: int, 
                        step: int, 
                        loss: float,
                        filename: Optional[str] = None) -> None:
        """Save checkpoint.
        
        Args:
            model: Model to save.
            optimizer: Optimizer to save.
            scheduler: Learning rate scheduler to save.
            epoch: Current epoch.
            step: Current step.
            loss: Current loss.
            filename: Optional filename for checkpoint.
        """
        # Create checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'config': self.config
        }
        
        # Save checkpoint
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}_step_{step}.pt"
        
        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        print(f"Checkpoint saved to {os.path.join(self.save_dir, filename)}")
