"""Unit test for Image Modeling task with Linear Transformer."""

import os
import torch
import unittest
import warnings
import logging

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress C++ extension warnings (for the ReduceOps.cpp warnings)
logging.getLogger("torch._C").setLevel(logging.ERROR)

# Set logging level to ERROR for all loggers
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.ERROR)

import sys
sys.path.append('/home/ubuntu/clean_code')

from datasets.image_modeling import ImageModelingDataset
from models.linear_transformer_image import LinearTransformerImageClassifier


class TestImageModeling(unittest.TestCase):
    """Test cases for Image Modeling task."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test config
        self.config = {
            'data_dir': 'data/imagenet1k_subset/',
            'image_size': 224,
            'patch_size': 16,
            'in_channels': 3,
            'normalize': True,
            'batch_size': 2,
            'num_workers': 0,
            'hidden_size': 256,
            'num_layers': 2,
            'num_heads': 4,
            'ff_size': 512,
            'dropout': 0.1,
            'num_classes': 1000,
            'device': 'cpu'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.config['data_dir'], exist_ok=True)
    
    def test_model_creation(self):
        """Test model creation with dummy input."""
        print("Testing model creation...")
        
        # Create model
        model = LinearTransformerImageClassifier(self.config)
        
        # Test forward pass with dummy input
        batch_size = 2
        image_size = self.config['image_size']
        in_channels = self.config['in_channels']
        
        # Create dummy input
        dummy_input = {
            'input_ids': torch.randn(batch_size, in_channels, image_size, image_size)
        }
        
        # Forward pass
        try:
            outputs = model(dummy_input)
            
            # Check output shape
            self.assertEqual(outputs['logits'].shape, (batch_size, self.config['num_classes']))
            print("Model creation test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Model creation test failed: {e}")
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        print("Testing dataset creation...")
        
        try:
            # Create dataset
            dataset = ImageModelingDataset(self.config)
            
            # Check if train, valid, and test datasets are created
            self.assertIsNotNone(dataset.train_dataset)
            self.assertIsNotNone(dataset.valid_dataset)
            self.assertIsNotNone(dataset.test_dataset)
            
            print("Dataset creation test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Dataset creation test failed: {e}")
    
    def test_dataloader_creation(self):
        """Test dataloader creation."""
        print("Testing dataloader creation...")
        
        try:
            # Create dataset
            dataset = ImageModelingDataset(self.config)
            
            # Create dataloaders
            train_loader = dataset.get_train_dataloader(self.config)
            valid_loader = dataset.get_valid_dataloader(self.config)
            test_loader = dataset.get_test_dataloader(self.config)
            
            # Check if dataloaders are created
            self.assertIsNotNone(train_loader)
            self.assertIsNotNone(valid_loader)
            self.assertIsNotNone(test_loader)
            
            print("Dataloader creation test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Dataloader creation test failed: {e}")
    
    def test_end_to_end(self):
        """Test end-to-end pipeline."""
        print("Testing end-to-end pipeline...")
        
        try:
            # Create dataset
            dataset = ImageModelingDataset(self.config)
            
            # Create model
            model = LinearTransformerImageClassifier(self.config)
            
            # Create dataloader
            train_loader = dataset.get_train_dataloader(self.config)
            
            # Get a batch
            for batch in train_loader:
                # Forward pass
                outputs = model(batch)
                
                # Compute loss
                loss = model.get_loss(outputs, batch)
                
                # Check if loss is a scalar
                self.assertEqual(loss.dim(), 0)
                
                # Only test one batch
                break
            
            print("End-to-end test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"End-to-end test failed: {e}")
    
    def test_different_input_formats(self):
        """Test handling of different input formats."""
        print("Testing different input formats...")
        
        # Create model
        model = LinearTransformerImageClassifier(self.config)
        
        # Test forward pass with different input formats
        batch_size = 2
        image_size = self.config['image_size']
        in_channels = self.config['in_channels']
        
        # Create dummy inputs in different formats
        # 1. Dictionary format
        dict_input = {
            'input_ids': torch.randn(batch_size, in_channels, image_size, image_size),
            'labels': torch.randint(0, self.config['num_classes'], (batch_size,))
        }
        
        # 2. Tuple format
        tuple_input = (
            torch.randn(batch_size, in_channels, image_size, image_size),
            torch.randint(0, self.config['num_classes'], (batch_size,))
        )
        
        # 3. Direct tensor format
        tensor_input = torch.randn(batch_size, in_channels, image_size, image_size)
        
        # Forward pass with different formats
        try:
            # Dictionary format
            dict_outputs = model(dict_input)
            self.assertEqual(dict_outputs['logits'].shape, (batch_size, self.config['num_classes']))
            
            # Tuple format
            tuple_outputs = model(tuple_input)
            self.assertEqual(tuple_outputs['logits'].shape, (batch_size, self.config['num_classes']))
            
            # Direct tensor format
            tensor_outputs = model(tensor_input)
            self.assertEqual(tensor_outputs['logits'].shape, (batch_size, self.config['num_classes']))
            
            print("Different input formats test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Different input formats test failed: {e}")
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        print("Testing empty batch handling...")
        
        # Create model
        model = LinearTransformerImageClassifier(self.config)
        
        # Test forward pass with empty batch
        batch_size = 0
        image_size = self.config['image_size']
        in_channels = self.config['in_channels']
        
        # Create dummy input with empty batch
        dummy_input = {
            'input_ids': torch.randn(batch_size, in_channels, image_size, image_size),
            'labels': torch.randint(0, self.config['num_classes'], (batch_size,))
        }
        
        # Forward pass
        try:
            outputs = model(dummy_input)
            
            # Check output shape
            self.assertEqual(outputs['logits'].shape[0], batch_size)
            self.assertEqual(outputs['logits'].shape[1], self.config['num_classes'])
            
            print("Empty batch handling test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Empty batch handling test failed: {e}")


if __name__ == '__main__':
    unittest.main()
