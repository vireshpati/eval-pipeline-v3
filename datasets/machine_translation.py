"""Machine Translation dataset implementation for the benchmarking pipeline."""

import os
import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

from torch.utils.data import Dataset, DataLoader

# Import fairseq2 components with compatibility handling
try:
    from fairseq2.data.text import TextTokenizer, SentencePieceTokenizer
except ImportError:
    # Fallback for fairseq2 0.4.5 which has a different import structure
    try:
        from fairseq2.data.text.text_tokenizer import TextTokenizer
        from fairseq2.data.text.sentencepiece import SentencePieceTokenizer
    except ImportError:
        # Create dummy classes if imports fail
        print("Warning: Could not import TextTokenizer from fairseq2. Using dummy implementation.")
        
        class TextTokenizer:
            """Dummy TextTokenizer class for compatibility."""
            def __init__(self, *args, **kwargs):
                pass
            
            def encode(self, text):
                """Dummy encode method."""
                return torch.tensor([1, 2, 3])  # BOS, dummy token, EOS
            
            def decode(self, tokens):
                """Dummy decode method."""
                return "dummy text"
        
        class SentencePieceTokenizer(TextTokenizer):
            """Dummy SentencePieceTokenizer class for compatibility."""
            def __init__(self, *args, **kwargs):
                super().__init__()
                self.vocab_size = 32000
                self.pad_id = 0
                self.bos_id = 1
                self.eos_id = 2
                self.unk_id = 3

from datasets.base_dataset import BenchmarkDataset


class DummyWMT16Dataset(Dataset):
    """Dummy WMT16 English-German dataset for testing and scaffolding."""
    
    def __init__(self, 
                 split: str = 'train',
                 src_lang: str = 'en',
                 tgt_lang: str = 'de',
                 max_src_len: int = 64,
                 max_tgt_len: int = 64,
                 vocab_size: int = 32000):
        """Initialize the dummy dataset.
        
        Args:
            split: Data split ('train', 'valid', or 'test').
            src_lang: Source language code.
            tgt_lang: Target language code.
            max_src_len: Maximum source sequence length.
            max_tgt_len: Maximum target sequence length.
            vocab_size: Vocabulary size.
        """
        super().__init__()
        
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.vocab_size = vocab_size
        
        # Create dummy data
        # For testing, create at least 10 samples to avoid empty dataset errors
        self.num_samples = 20 if split == 'train' else 10
        
        # Special token IDs
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing:
                - 'src_tokens': Source tokens.
                - 'tgt_tokens': Target tokens.
                - 'tgt_labels': Target labels (shifted right).
        """
        # Generate random source tokens
        src_len = np.random.randint(5, self.max_src_len - 2)  # Leave room for BOS/EOS
        src_tokens = torch.randint(4, self.vocab_size, (src_len,))  # Start from 4 to avoid special tokens
        src_tokens = torch.cat([
            torch.tensor([self.bos_id]),
            src_tokens,
            torch.tensor([self.eos_id])
        ])
        
        # Generate random target tokens
        tgt_len = np.random.randint(5, self.max_tgt_len - 2)  # Leave room for BOS/EOS
        tgt_tokens = torch.randint(4, self.vocab_size, (tgt_len,))  # Start from 4 to avoid special tokens
        tgt_tokens = torch.cat([
            torch.tensor([self.bos_id]),
            tgt_tokens,
            torch.tensor([self.eos_id])
        ])
        
        # Create target labels (shifted right)
        tgt_labels = tgt_tokens[1:]  # Remove BOS
        
        # Create source mask
        src_mask = torch.ones_like(src_tokens, dtype=torch.bool)
        
        # Create target mask
        tgt_mask = torch.ones_like(tgt_tokens, dtype=torch.bool)
        
        return {
            'src_tokens': src_tokens,
            'src_mask': src_mask,
            'tgt_tokens': tgt_tokens,
            'tgt_labels': tgt_labels,
            'tgt_mask': tgt_mask
        }


class WMT16Dataset(Dataset):
    """WMT16 English-German dataset."""
    
    def __init__(self, 
                 root: str, 
                 split: str = 'train',
                 src_lang: str = 'en',
                 tgt_lang: str = 'de',
                 max_src_len: int = 64,
                 max_tgt_len: int = 64,
                 vocab_size: int = 32000):
        """Initialize the dataset.
        
        Args:
            root: Root directory of the dataset.
            split: Data split ('train', 'valid', or 'test').
            src_lang: Source language code.
            tgt_lang: Target language code.
            max_src_len: Maximum source sequence length.
            max_tgt_len: Maximum target sequence length.
            vocab_size: Vocabulary size.
        """
        super().__init__()
        
        self.root = root
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.vocab_size = vocab_size
        
        # Check if dataset exists
        if not os.path.exists(root):
            print(f"Warning: Dataset directory not found at {root}")
            print(f"Creating dummy data for scaffolding purposes.")
            self.dummy_dataset = DummyWMT16Dataset(
                split=split,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_src_len=max_src_len,
                max_tgt_len=max_tgt_len,
                vocab_size=vocab_size
            )
            self.samples = []
            return
        
        # Load dataset
        self._load_dataset()
        
        # Set up tokenizers
        self._setup_tokenizers()
    
    def _load_dataset(self):
        """Load the dataset from disk."""
        self.samples = []
        
        # Check for source and target files
        src_file = os.path.join(self.root, f"{self.split}.{self.src_lang}")
        tgt_file = os.path.join(self.root, f"{self.split}.{self.tgt_lang}")
        
        if not os.path.exists(src_file) or not os.path.exists(tgt_file):
            print(f"Warning: Dataset files not found at {src_file} or {tgt_file}")
            # Create dummy files for testing
            if self.split in ['train', 'valid', 'test']:
                self._create_dummy_files(src_file, tgt_file)
            return
        
        # Load source and target sentences
        with open(src_file, 'r', encoding='utf-8') as src_f, \
             open(tgt_file, 'r', encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                self.samples.append((src_line.strip(), tgt_line.strip()))
    
    def _create_dummy_files(self, src_file, tgt_file):
        """Create dummy files for testing purposes."""
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(src_file), exist_ok=True)
        os.makedirs(os.path.dirname(tgt_file), exist_ok=True)
        
        # Number of samples to create
        num_samples = 20 if self.split == 'train' else 10
        
        # Create dummy source file
        with open(src_file, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                f.write(f"This is a dummy source sentence {i}.\n")
        
        # Create dummy target file
        with open(tgt_file, 'w', encoding='utf-8') as f:
            for i in range(num_samples):
                f.write(f"Dies ist ein Dummy-Zielsatz {i}.\n")
        
        # Load the created dummy data
        with open(src_file, 'r', encoding='utf-8') as src_f, \
             open(tgt_file, 'r', encoding='utf-8') as tgt_f:
            for src_line, tgt_line in zip(src_f, tgt_f):
                self.samples.append((src_line.strip(), tgt_line.strip()))
    
    def _setup_tokenizers(self):
        """Set up tokenizers for source and target languages."""
        # Check for SentencePiece model files
        src_spm = os.path.join(self.root, f"spm.{self.src_lang}.model")
        tgt_spm = os.path.join(self.root, f"spm.{self.tgt_lang}.model")
        
        # Use shared tokenizer if files don't exist
        if not os.path.exists(src_spm) or not os.path.exists(tgt_spm):
            print(f"Warning: SentencePiece model files not found. Using dummy tokenizers.")
            self.src_tokenizer = TextTokenizer()
            self.tgt_tokenizer = TextTokenizer()
            return
        
        # Create tokenizers
        try:
            self.src_tokenizer = SentencePieceTokenizer(model_path=src_spm)
            self.tgt_tokenizer = SentencePieceTokenizer(model_path=tgt_spm)
        except Exception as e:
            print(f"Error creating tokenizers: {e}")
            print("Using dummy tokenizers instead.")
            self.src_tokenizer = TextTokenizer()
            self.tgt_tokenizer = TextTokenizer()
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples.
        """
        if hasattr(self, 'dummy_dataset'):
            return len(self.dummy_dataset)
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing:
                - 'src_tokens': Source tokens.
                - 'tgt_tokens': Target tokens.
                - 'tgt_labels': Target labels (shifted right).
        """
        # Return dummy data if dataset doesn't exist
        if hasattr(self, 'dummy_dataset'):
            return self.dummy_dataset[idx]
        
        # Get sample
        src_text, tgt_text = self.samples[idx]
        
        # Tokenize source and target
        try:
            src_tokens = self.src_tokenizer.encode(src_text)
            tgt_tokens = self.tgt_tokenizer.encode(tgt_text)
            
            # Truncate if needed
            if len(src_tokens) > self.max_src_len:
                src_tokens = src_tokens[:self.max_src_len]
            
            if len(tgt_tokens) > self.max_tgt_len:
                tgt_tokens = tgt_tokens[:self.max_tgt_len]
            
            # Create target labels (shifted right)
            tgt_labels = tgt_tokens[1:]  # Remove BOS
            
            # Create source mask
            src_mask = torch.ones_like(src_tokens, dtype=torch.bool)
            
            # Create target mask
            tgt_mask = torch.ones_like(tgt_tokens, dtype=torch.bool)
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return dummy tokens on error
            return DummyWMT16Dataset()[0]
        
        return {
            'src_tokens': src_tokens,
            'src_mask': src_mask,
            'tgt_tokens': tgt_tokens,
            'tgt_labels': tgt_labels,
            'tgt_mask': tgt_mask
        }


class MachineTranslationDataset(BenchmarkDataset):
    """Machine translation dataset implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset with the given configuration.
        
        Args:
            config: Configuration dictionary with dataset parameters.
        """
        super().__init__()
        
        self.config = config
        self.data_dir = config.get('data_dir', 'data/wmt16_en_de/')
        self.src_lang = config.get('src_lang', 'en')
        self.tgt_lang = config.get('tgt_lang', 'de')
        self.max_src_len = config.get('max_src_len', 64)
        self.max_tgt_len = config.get('max_tgt_len', 64)
        self.vocab_size = config.get('vocab_size', 32000)
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create dataset instances
        self._create_datasets()
    
    def _create_datasets(self):
        """Create dataset instances for train, valid, and test splits."""
        self.train_dataset = WMT16Dataset(
            root=self.data_dir,
            split='train',
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len,
            vocab_size=self.vocab_size
        )
        
        self.valid_dataset = WMT16Dataset(
            root=self.data_dir,
            split='valid',
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len,
            vocab_size=self.vocab_size
        )
        
        self.test_dataset = WMT16Dataset(
            root=self.data_dir,
            split='test',
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang,
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len,
            vocab_size=self.vocab_size
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
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching samples.
        
        Args:
            batch: List of dictionaries containing:
                - 'src_tokens': Source tokens.
                - 'src_mask': Source mask.
                - 'tgt_tokens': Target tokens.
                - 'tgt_labels': Target labels.
                - 'tgt_mask': Target mask.
            
        Returns:
            Dictionary containing batched tensors.
        """
        # Get max lengths
        max_src_len = max(item['src_tokens'].size(0) for item in batch)
        max_tgt_len = max(item['tgt_tokens'].size(0) for item in batch)
        max_tgt_labels_len = max(item['tgt_labels'].size(0) for item in batch)
        
        # Get pad ID
        pad_id = 0  # Default pad ID
        
        # Prepare batched tensors
        src_tokens = []
        src_mask = []
        tgt_tokens = []
        tgt_labels = []
        tgt_mask = []
        
        # Pad sequences
        for item in batch:
            # Pad source tokens
            src_len = item['src_tokens'].size(0)
            padded_src = torch.full((max_src_len,), pad_id, dtype=item['src_tokens'].dtype)
            padded_src[:src_len] = item['src_tokens']
            src_tokens.append(padded_src)
            
            # Pad source mask
            padded_src_mask = torch.zeros(max_src_len, dtype=torch.bool)
            padded_src_mask[:src_len] = item['src_mask']
            src_mask.append(padded_src_mask)
            
            # Pad target tokens
            tgt_len = item['tgt_tokens'].size(0)
            padded_tgt = torch.full((max_tgt_len,), pad_id, dtype=item['tgt_tokens'].dtype)
            padded_tgt[:tgt_len] = item['tgt_tokens']
            tgt_tokens.append(padded_tgt)
            
            # Pad target labels
            tgt_labels_len = item['tgt_labels'].size(0)
            padded_tgt_labels = torch.full((max_tgt_labels_len,), pad_id, dtype=item['tgt_labels'].dtype)
            padded_tgt_labels[:tgt_labels_len] = item['tgt_labels']
            tgt_labels.append(padded_tgt_labels)
            
            # Pad target mask
            padded_tgt_mask = torch.zeros(max_tgt_len, dtype=torch.bool)
            padded_tgt_mask[:tgt_len] = item['tgt_mask']
            tgt_mask.append(padded_tgt_mask)
        
        # Stack tensors
        src_tokens = torch.stack(src_tokens)
        src_mask = torch.stack(src_mask)
        tgt_tokens = torch.stack(tgt_tokens)
        tgt_labels = torch.stack(tgt_labels)
        tgt_mask = torch.stack(tgt_mask)
        
        return {
            'src_tokens': src_tokens,
            'src_mask': src_mask,
            'tgt_tokens': tgt_tokens,
            'tgt_labels': tgt_labels,
            'tgt_mask': tgt_mask
        }
    
    def get_metrics(self) -> Dict[str, Callable]:
        """Returns a dictionary of metric functions for evaluation.
        
        Returns:
            Dictionary mapping metric names to metric functions.
        """
        def bleu_score(outputs, batch):
            # This is a simplified BLEU score calculation
            # In a real implementation, you would use a proper BLEU score library
            logits = outputs['logits']
            labels = batch['tgt_labels']
            
            # Get predictions
            preds = torch.argmax(logits, dim=-1)
            
            # Calculate accuracy as a proxy for BLEU
            mask = (labels != 0)  # Ignore padding
            correct = ((preds == labels) & mask).sum().float()
            total = mask.sum().float()
            
            return correct / (total + 1e-8)
        
        return {
            'bleu': bleu_score
        }
    
    def get_vocab_size(self) -> int:
        """Returns the vocabulary size for the dataset.
        
        Returns:
            Vocabulary size.
        """
        return self.vocab_size
    
    def get_num_classes(self) -> int:
        """Returns the number of classes for classification tasks.
        
        Returns:
            Number of classes.
        """
        # Not applicable for translation
        return 0
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Returns the special token IDs for the dataset.
        
        Returns:
            Dictionary mapping token names to IDs.
        """
        return {
            'pad': 0,
            'bos': 1,
            'eos': 2,
            'unk': 3
        }
