"""Configuration utilities for the benchmarking pipeline."""

import os
import yaml
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary.
        config_path: Path to save configuration file.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary.
        override_config: Override configuration dictionary.
        
    Returns:
        Merged configuration dictionary.
    """
    merged_config = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged_config and isinstance(merged_config[key], dict):
            merged_config[key] = merge_configs(merged_config[key], value)
        else:
            merged_config[key] = value
    
    return merged_config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration.
    
    Returns:
        Default configuration dictionary.
    """
    return {
        'dataset': {
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True
        },
        'model': {
            'hidden_size': 768,
            'num_layers': 6,
            'num_heads': 8,
            'ff_size': 2048,
            'dropout': 0.1
        },
        'trainer': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 10,
            'device': 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu',
            'log_every_n_steps': 10,
            'save_every_n_steps': 100,
            'save_dir': 'checkpoints/',
            'mixed_precision': False
        }
    }
