"""Test script for speech classification task."""

import os
import warnings
import logging
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress C++ extension warnings (for the ReduceOps.cpp warnings)
logging.getLogger("torch._C").setLevel(logging.ERROR)
os.environ["TORCH_CPP_LOG_LEVEL"] = "50"  # Critical level only
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

# Disable all logging below ERROR level
logging.basicConfig(level=logging.ERROR)

# Import local modules
from datasets.speech_classification import SpeechClassificationDataset, DummySC10Dataset
from models.linear_transformer import LinearTransformerClassifier
from trainer.trainer import Trainer


def test_dataset_creation():
    """Test creation of speech classification dataset."""
    try:
        print("\n=== Testing dataset creation ===")
        
        # Create dataset config
        config = {
            'data_dir': 'data/sc10/',
            'n_mels': 80,
            'sample_rate': 16000
        }
        
        # Create dataset
        dataset = SpeechClassificationDataset(config)
        
        # Check if dataset was created successfully
        print(f"Train dataset size: {len(dataset.train_dataset)}")
        print(f"Valid dataset size: {len(dataset.valid_dataset)}")
        print(f"Test dataset size: {len(dataset.test_dataset)}")
        
        # Verify dataset is not empty
        assert len(dataset.train_dataset) > 0, "Train dataset is empty"
        assert len(dataset.valid_dataset) > 0, "Valid dataset is empty"
        assert len(dataset.test_dataset) > 0, "Test dataset is empty"
        
        # Get a sample from the dataset
        sample, label = dataset.train_dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Label: {label}")
        
        # Verify sample shape
        assert sample.dim() == 2, f"Expected 2D tensor, got {sample.dim()}D"
        assert sample.size(0) == config['n_mels'], f"Expected {config['n_mels']} mel bands, got {sample.size(0)}"
        
        print("Dataset creation test passed!")
        return True
    
    except Exception as e:
        print(f"Dataset creation test failed: {e}")
        traceback.print_exc()
        return False


def test_model_creation():
    """Test creation of linear transformer model for speech classification."""
    try:
        print("\n=== Testing model creation ===")
        
        # Create model config
        config = {
            'num_classes': 10,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'ff_size': 1024,
            'dropout': 0.1,
            'n_mels': 80
        }
        
        # Create model
        model = LinearTransformerClassifier(config)
        
        # Check model name
        print(f"Model name: {model.name}")
        
        # Test forward pass with dummy input
        batch_size = 2
        seq_len = 32  # Match the sequence length used in the model's position embeddings
        n_mels = 80   # Match the n_mels in config
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, n_mels)
        dummy_labels = torch.randint(0, config['num_classes'], (batch_size,))
        
        # Create batch
        batch = {
            'input_ids': dummy_input,
            'labels': dummy_labels
        }
        
        # Forward pass
        outputs = model(batch)
        
        # Check outputs
        print(f"Logits shape: {outputs['logits'].shape}")
        print(f"Embeddings shape: {outputs['embeddings'].shape}")
        
        # Verify output shapes
        assert outputs['logits'].shape == (batch_size, config['num_classes']), \
            f"Expected logits shape {(batch_size, config['num_classes'])}, got {outputs['logits'].shape}"
        assert outputs['embeddings'].shape == (batch_size, config['hidden_size']), \
            f"Expected embeddings shape {(batch_size, config['hidden_size'])}, got {outputs['embeddings'].shape}"
        
        # Test loss computation
        loss = model.get_loss(outputs, batch)
        print(f"Loss: {loss.item()}")
        
        # Verify loss is a scalar
        assert loss.dim() == 0, f"Expected scalar loss, got tensor with {loss.dim()} dimensions"
        
        print("Model creation test passed!")
        return True
    
    except Exception as e:
        print(f"Model creation test failed: {e}")
        traceback.print_exc()
        return False


def test_dataloader_creation():
    """Test creation of dataloaders for speech classification."""
    try:
        print("\n=== Testing dataloader creation ===")
        
        # Create dataset config
        dataset_config = {
            'data_dir': 'data/sc10/',
            'n_mels': 80,
            'sample_rate': 16000
        }
        
        # Create dataset
        dataset = SpeechClassificationDataset(dataset_config)
        
        # Create dataloader config
        dataloader_config = {
            'batch_size': 4,
            'num_workers': 0
        }
        
        # Create dataloaders
        train_dataloader = dataset.get_train_dataloader(dataloader_config)
        valid_dataloader = dataset.get_valid_dataloader(dataloader_config)
        test_dataloader = dataset.get_test_dataloader(dataloader_config)
        
        # Check dataloaders
        print(f"Train dataloader length: {len(train_dataloader)}")
        print(f"Valid dataloader length: {len(valid_dataloader)}")
        print(f"Test dataloader length: {len(test_dataloader)}")
        
        # Get a batch from the train dataloader
        batch = next(iter(train_dataloader))
        
        # Print batch structure
        print("Batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # Verify batch structure
        assert 'input_ids' in batch, "Batch missing 'input_ids'"
        assert 'labels' in batch, "Batch missing 'labels'"
        
        # Verify batch shapes
        assert batch['input_ids'].dim() == 3, f"Expected 3D tensor for input_ids, got {batch['input_ids'].dim()}D"
        assert batch['labels'].dim() == 1, f"Expected 1D tensor for labels, got {batch['labels'].dim()}D"
        
        print("Dataloader creation test passed!")
        return True
    
    except Exception as e:
        print(f"Dataloader creation test failed: {e}")
        traceback.print_exc()
        return False


def test_audio_processing():
    """Test audio processing for speech classification."""
    try:
        print("\n=== Testing audio processing ===")
        
        # Create dummy dataset
        dummy_dataset = DummySC10Dataset(
            split='train',
            n_mels=80,
            sample_rate=16000
        )
        
        # Get a sample from the dataset
        features, label = dummy_dataset[0]
        
        # Check feature shape
        print(f"Feature shape: {features.shape}")
        print(f"Label: {label}")
        
        # Verify feature shape
        assert features.dim() == 2, f"Expected 2D tensor, got {features.dim()}D"
        assert features.size(0) == 80, f"Expected 80 mel bands, got {features.size(0)}"
        
        # Create a batch of samples
        batch_size = 4
        features_list = []
        labels_list = []
        
        for i in range(batch_size):
            feat, lab = dummy_dataset[i]
            features_list.append(feat)
            labels_list.append(lab)
        
        # Get max sequence length
        max_length = max(feat.size(1) for feat in features_list)
        
        # Pad features to max length
        padded_features = []
        for feat in features_list:
            # [n_mels, time] -> [n_mels, max_length]
            padded_feat = torch.zeros(80, max_length, dtype=feat.dtype)
            padded_feat[:, :feat.size(1)] = feat
            padded_features.append(padded_feat)
        
        # Stack tensors
        # [batch_size, n_mels, max_length] -> [batch_size, max_length, n_mels]
        features_tensor = torch.stack(padded_features).transpose(1, 2)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)
        
        # Check batch shapes
        print(f"Batch features shape: {features_tensor.shape}")
        print(f"Batch labels shape: {labels_tensor.shape}")
        
        # Verify batch shapes
        assert features_tensor.shape == (batch_size, max_length, 80), \
            f"Expected shape {(batch_size, max_length, 80)}, got {features_tensor.shape}"
        assert labels_tensor.shape == (batch_size,), \
            f"Expected shape {(batch_size,)}, got {labels_tensor.shape}"
        
        print("Audio processing test passed!")
        return True
    
    except Exception as e:
        print(f"Audio processing test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end():
    """Test end-to-end training for speech classification."""
    try:
        print("\n=== Testing end-to-end training ===")
        
        # Create dataset config
        dataset_config = {
            'data_dir': 'data/sc10/',
            'n_mels': 80,
            'sample_rate': 16000
        }
        
        # Create model config
        model_config = {
            'num_classes': 10,
            'hidden_size': 256,
            'num_layers': 4,
            'num_heads': 4,
            'ff_size': 1024,
            'dropout': 0.1,
            'n_mels': 80
        }
        
        # Create trainer config
        trainer_config = {
            'batch_size': 4,
            'num_workers': 0,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 1,
            'device': 'cpu',
            'log_every_n_steps': 1,
            'save_every_n_steps': 10,
            'save_dir': 'checkpoints/',
            'mixed_precision': False
        }
        
        # Create dataset
        dataset = SpeechClassificationDataset(dataset_config)
        
        # Create model
        model = LinearTransformerClassifier(model_config)
        
        # Create trainer
        trainer = Trainer(trainer_config)
        
        # Train for 1 step
        trainer_config['max_steps'] = 1
        
        # Train model
        trainer.train(
            model=model,
            train_dataloader=dataset.get_train_dataloader(trainer_config),
            valid_dataloader=dataset.get_valid_dataloader(trainer_config),
            metrics=dataset.get_metrics()
        )
        
        # Evaluate model
        metrics = trainer.evaluate(
            model=model,
            dataloader=dataset.get_test_dataloader(trainer_config),
            metrics=dataset.get_metrics()
        )
        
        # Check metrics
        print(f"Evaluation metrics: {metrics}")
        
        # Verify metrics
        assert 'accuracy' in metrics, "Metrics missing 'accuracy'"
        
        print("End-to-end training test passed!")
        return True
    
    except Exception as e:
        print(f"End-to-end training test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Running speech classification tests...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data/sc10/', exist_ok=True)
    
    # Run tests
    tests = [
        test_dataset_creation,
        test_model_creation,
        test_dataloader_creation,
        test_audio_processing,
        test_end_to_end
    ]
    
    # Track test results
    results = []
    
    # Run each test
    for test in tests:
        result = test()
        results.append(result)
    
    # Print summary
    print("\n=== Test Summary ===")
    for i, (test, result) in enumerate(zip(tests, results)):
        print(f"{i+1}. {test.__name__}: {'PASSED' if result else 'FAILED'}")
    
    # Check if all tests passed
    if all(results):
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    main()
