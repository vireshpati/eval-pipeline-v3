"""Main script for training and evaluating speech classification models."""

import argparse
import os
import sys
import torch

from datasets.speech_classification import SpeechClassificationDataset
from models.linear_transformer import LinearTransformerClassifier
from trainer.config import load_config
from trainer.trainer import BenchmarkTrainer
from trainer.utils import set_seed, log_model_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Speech Classification Benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--checkpoint", type=str, default=None, 
                        help="Path to checkpoint for evaluation")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory from config")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed from config")
    parser.add_argument("--fp16", action="store_true", default=None,
                        help="Override FP16 setting from config")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from config")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_dir is not None:
        config["dataset"]["data_dir"] = args.data_dir
    
    if args.output_dir is not None:
        config["checkpoint_dir"] = args.output_dir
    
    if args.seed is not None:
        config["seed"] = args.seed
    
    if args.fp16 is not None:
        config["fp16"] = args.fp16
    
    if args.device is not None:
        config["device"] = args.device
    
    # Set random seed
    set_seed(config["seed"])
    
    # Create dataset
    print(f"Creating dataset: {config['task']}")
    dataset = SpeechClassificationDataset(config["dataset"])
    
    # Update model config with dataset info
    config["model"]["num_classes"] = dataset.get_num_classes()
    
    # Create model
    print(f"Creating model: {config['model']['name']}")
    model = LinearTransformerClassifier(config["model"])
    
    # Log model information
    log_model_info(model)
    
    # Create trainer
    trainer = BenchmarkTrainer(model, dataset, config)
    
    # Train or evaluate
    if args.mode == "train":
        print("Starting training...")
        trainer.train()
    else:
        if args.checkpoint is None:
            print("Error: Checkpoint path is required for evaluation mode")
            sys.exit(1)
        
        print(f"Loading checkpoint: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
        
        print("Evaluating on test set...")
        test_metrics = trainer.evaluate("test")
        
        print("Test results:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
