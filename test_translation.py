"""Unit test for Machine Translation task with Linear Transformer."""

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

from datasets.machine_translation import MachineTranslationDataset
from models.linear_transformer_translator import LinearTransformerTranslator


class TestMachineTranslation(unittest.TestCase):
    """Test cases for Machine Translation task."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test config
        self.config = {
            'data_dir': 'data/wmt16_en_de/',
            'src_lang': 'en',
            'tgt_lang': 'de',
            'max_src_len': 64,
            'max_tgt_len': 64,
            'vocab_size': 32000,
            'batch_size': 2,
            'num_workers': 0,
            'hidden_size': 256,
            'num_layers': 2,
            'num_heads': 4,
            'ff_size': 512,
            'dropout': 0.1,
            'src_vocab_size': 32000,
            'tgt_vocab_size': 32000,
            'share_embeddings': True,
            'pad_idx': 0,
            'bos_idx': 1,
            'eos_idx': 2,
            'device': 'cpu'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.config['data_dir'], exist_ok=True)
    
    def test_model_creation(self):
        """Test model creation with dummy input."""
        print("Testing model creation...")
        
        # Create model
        model = LinearTransformerTranslator(self.config)
        
        # Test forward pass with dummy input
        batch_size = 2
        src_len = 32  # Match the sequence length used in the model's position embeddings
        tgt_len = 32  # Match the sequence length used in the model's position embeddings
        
        # Create dummy input
        dummy_input = {
            'src_tokens': torch.randint(0, self.config['src_vocab_size'], (batch_size, src_len)),
            'src_mask': torch.ones(batch_size, src_len, dtype=torch.bool),
            'tgt_tokens': torch.randint(0, self.config['tgt_vocab_size'], (batch_size, tgt_len)),
            'tgt_labels': torch.randint(0, self.config['tgt_vocab_size'], (batch_size, tgt_len)),
            'tgt_mask': torch.ones(batch_size, tgt_len, dtype=torch.bool)
        }
        
        # Forward pass
        try:
            outputs = model(dummy_input)
            
            # Check output shape
            self.assertEqual(outputs['logits'].shape, (batch_size, tgt_len, self.config['tgt_vocab_size']))
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
            dataset = MachineTranslationDataset(self.config)
            
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
            dataset = MachineTranslationDataset(self.config)
            
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
            dataset = MachineTranslationDataset(self.config)
            
            # Create model
            model = LinearTransformerTranslator(self.config)
            
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
    
    def test_empty_sequence_handling(self):
        """Test handling of empty sequences."""
        print("Testing empty sequence handling...")
        
        # Create model
        model = LinearTransformerTranslator(self.config)
        
        # Test forward pass with empty input
        batch_size = 2
        
        # Create dummy input with empty sequences
        dummy_input = {
            'src_tokens': torch.zeros((batch_size, 0), dtype=torch.long),
            'src_mask': torch.zeros((batch_size, 0), dtype=torch.bool),
            'tgt_tokens': torch.zeros((batch_size, 0), dtype=torch.long),
            'tgt_labels': torch.zeros((batch_size, 0), dtype=torch.long),
            'tgt_mask': torch.zeros((batch_size, 0), dtype=torch.bool)
        }
        
        # Forward pass
        try:
            outputs = model(dummy_input)
            
            # Check that we get outputs without errors
            self.assertIn('logits', outputs)
            self.assertIn('encoder_out', outputs)
            self.assertIn('decoder_out', outputs)
            
            print("Empty sequence handling test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Empty sequence handling test failed: {e}")
    
    def test_generation(self):
        """Test generation functionality."""
        print("Testing generation functionality...")
        
        # Create model
        model = LinearTransformerTranslator(self.config)
        
        # Test generation with dummy input
        batch_size = 2
        src_len = 32
        
        # Create dummy input
        dummy_input = {
            'src_tokens': torch.randint(0, self.config['src_vocab_size'], (batch_size, src_len)),
            'src_mask': torch.ones(batch_size, src_len, dtype=torch.bool)
        }
        
        # Generation
        try:
            generated = model.generate(dummy_input, max_len=10)
            
            # Check output shape
            self.assertEqual(generated.dim(), 2)
            self.assertEqual(generated.size(0), batch_size)
            self.assertGreaterEqual(generated.size(1), 1)  # At least BOS token
            
            print("Generation test passed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.fail(f"Generation test failed: {e}")


if __name__ == '__main__':
    unittest.main()
