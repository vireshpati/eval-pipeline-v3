"""Linear Transformer implementation for machine translation."""

from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BenchmarkModel
from models.linear_transformer import LinearTransformerEncoder


class LinearTransformerTranslator(BenchmarkModel):
    """Linear Transformer model for translation tasks."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model with the given configuration.
        
        Args:
            config: Configuration dictionary with model parameters.
        """
        super().__init__(config)
        
        # Model configuration
        self.vocab_size = config.get('vocab_size', 32000)
        self.hidden_size = config.get('hidden_size', 768)
        self.num_layers = config.get('num_layers', 6)
        self.num_heads = config.get('num_heads', 8)
        self.ff_size = config.get('ff_size', 2048)
        self.dropout = config.get('dropout', 0.1)
        self.max_seq_len = config.get('max_seq_len', 128)
        
        # Special token IDs
        self.pad_id = config.get('pad_id', 0)
        self.bos_id = config.get('bos_id', 1)
        self.eos_id = config.get('eos_id', 2)
        
        # Embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.hidden_size,
            padding_idx=self.pad_id
        )
        
        # Position embeddings - only create for non-zero sequence lengths
        try:
            self.register_buffer(
                "position_embedding", 
                torch.zeros(1, 1, self.hidden_size),
                persistent=False
            )
        except TypeError:
            # For older PyTorch versions that don't support persistent flag
            self.register_buffer(
                "position_embedding", 
                torch.zeros(1, 1, self.hidden_size)
            )
        
        # Encoder
        self.encoder = LinearTransformerEncoder(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            ff_size=self.ff_size,
            dropout=self.dropout
        )
        
        # Decoder (using the same architecture as encoder for simplicity)
        self.decoder = LinearTransformerEncoder(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            ff_size=self.ff_size,
            dropout=self.dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Layer normalization
        self.encoder_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        self.decoder_norm = nn.LayerNorm(self.hidden_size, eps=1e-5)
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        
        # Initialize parameters
        self._init_parameters()
    
    @property
    def name(self) -> str:
        """Return the name of the model.
        
        Returns:
            Model name.
        """
        return "linear_transformer_translator"
    
    def _init_parameters(self):
        """Initialize model parameters."""
        # Initialize token embedding
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.constant_(self.output_projection.bias, 0.0)
    
    def _get_position_embeddings(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get position embeddings of the required length.
        
        This method lazily initializes position embeddings when needed,
        avoiding issues with fixed buffer sizes during model creation.
        
        Args:
            seq_len: Sequence length needed
            device: Device to create embeddings on
            
        Returns:
            Position embeddings of shape [1, seq_len, hidden_size]
        """
        # For empty sequences, return empty tensor
        if seq_len == 0:
            return torch.zeros(1, 0, self.hidden_size, device=device)
            
        # If we need more positions than we have, create new ones
        if seq_len > self.position_embedding.size(1):
            # Create new position embeddings
            new_embeddings = torch.zeros(1, seq_len, self.hidden_size, device=device)
            
            # Initialize with normal distribution
            nn.init.normal_(new_embeddings, mean=0.0, std=0.02)
            
            # Update buffer
            try:
                self.register_buffer("position_embedding", new_embeddings, persistent=False)
            except TypeError:
                # For older PyTorch versions that don't support persistent flag
                self.register_buffer("position_embedding", new_embeddings)
            return new_embeddings
        
        # Otherwise return the slice we need
        return self.position_embedding[:, :seq_len, :]
    
    def forward(self, batch: Any) -> Dict[str, torch.Tensor]:
        """Forward pass of the model.
        
        Args:
            batch: Input batch, can be:
                - Dictionary containing 'src_tokens', 'src_mask', 'tgt_tokens', 'tgt_mask'
                - Tuple where first element is source tokens and second is target tokens
            
        Returns:
            Dictionary containing output tensors.
                - 'logits': Tensor of shape [batch_size, tgt_len, vocab_size].
                - 'encoder_out': Tensor of shape [batch_size, src_len, hidden_size].
                - 'decoder_out': Tensor of shape [batch_size, tgt_len, hidden_size].
        """
        # Handle different input types for maximum compatibility
        if isinstance(batch, dict):
            src_tokens = batch['src_tokens']
            src_mask = batch.get('src_mask', None)
            tgt_tokens = batch.get('tgt_tokens', None)
            tgt_mask = batch.get('tgt_mask', None)
        elif isinstance(batch, tuple):
            src_tokens = batch[0]
            tgt_tokens = batch[1] if len(batch) > 1 else None
            src_mask = None
            tgt_mask = None
        else:
            src_tokens = batch
            tgt_tokens = None
            src_mask = None
            tgt_mask = None
        
        # Get batch size and sequence lengths
        batch_size = src_tokens.size(0)
        src_len = src_tokens.size(1)
        device = src_tokens.device
        
        # Handle empty batch
        if batch_size == 0:
            return {
                'logits': torch.zeros(0, 0, self.vocab_size, device=device),
                'encoder_out': torch.zeros(0, 0, self.hidden_size, device=device),
                'decoder_out': torch.zeros(0, 0, self.hidden_size, device=device)
            }
        
        # Handle empty source sequences
        if src_len == 0:
            return {
                'logits': torch.zeros(batch_size, 0, self.vocab_size, device=device),
                'encoder_out': torch.zeros(batch_size, 0, self.hidden_size, device=device),
                'decoder_out': torch.zeros(batch_size, 0, self.hidden_size, device=device)
            }
        
        # Create source mask if not provided
        if src_mask is None:
            src_mask = (src_tokens != self.pad_id)
        
        # Embed source tokens
        src_emb = self.token_embedding(src_tokens)
        
        # Add position embeddings to source
        src_pos_emb = self._get_position_embeddings(src_len, device)
        src_emb = src_emb + src_pos_emb
        
        # Encode source
        encoder_out = self.encoder(src_emb, src_mask)
        encoder_out = self.encoder_norm(encoder_out)
        
        # If no target tokens provided, return encoder output only
        if tgt_tokens is None:
            return {
                'logits': torch.zeros(batch_size, 0, self.vocab_size, device=device),
                'encoder_out': encoder_out,
                'decoder_out': torch.zeros(batch_size, 0, self.hidden_size, device=device)
            }
        
        # Get target sequence length
        tgt_len = tgt_tokens.size(1)
        
        # Handle empty target sequences
        if tgt_len == 0:
            return {
                'logits': torch.zeros(batch_size, 0, self.vocab_size, device=device),
                'encoder_out': encoder_out,
                'decoder_out': torch.zeros(batch_size, 0, self.hidden_size, device=device)
            }
        
        # Create target mask if not provided
        if tgt_mask is None:
            tgt_mask = (tgt_tokens != self.pad_id)
        
        # Embed target tokens
        tgt_emb = self.token_embedding(tgt_tokens)
        
        # Add position embeddings to target
        tgt_pos_emb = self._get_position_embeddings(tgt_len, device)
        tgt_emb = tgt_emb + tgt_pos_emb
        
        # Decode target with encoder output
        # For simplicity, we're using the encoder architecture for decoding
        # In a real implementation, this would be a proper decoder with cross-attention
        decoder_out = self.decoder(tgt_emb, tgt_mask)
        decoder_out = self.decoder_norm(decoder_out)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_out)
        
        return {
            'logits': logits,
            'encoder_out': encoder_out,
            'decoder_out': decoder_out
        }
    
    def get_loss(self, outputs: Dict[str, torch.Tensor], batch: Any) -> torch.Tensor:
        """Compute loss for the given outputs and batch.
        
        Args:
            outputs: Model outputs from forward pass.
            batch: Input batch.
            
        Returns:
            Loss tensor.
        """
        # Get logits
        logits = outputs['logits']
        
        # Get target labels from batch
        if isinstance(batch, dict):
            tgt_labels = batch.get('tgt_labels', None)
        elif isinstance(batch, tuple) and len(batch) > 2:
            tgt_labels = batch[2]
        else:
            # If no labels provided, return zero loss
            return torch.tensor(0.0, device=logits.device)
        
        # Reshape logits for loss computation
        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        logits = logits.reshape(-1, logits.size(-1))
        
        # Reshape labels
        # [batch_size, seq_len] -> [batch_size * seq_len]
        tgt_labels = tgt_labels.reshape(-1)
        
        # Compute loss
        loss = self.loss_fn(logits, tgt_labels)
        
        return loss
    
    def generate(self, batch: Any, max_len: int = 100, beam_size: int = 1) -> torch.Tensor:
        """Generate translations for the given batch.
        
        Args:
            batch: Input batch.
            max_len: Maximum length of generated sequence.
            beam_size: Beam size for beam search.
            
        Returns:
            Generated token IDs of shape [batch_size, seq_len].
        """
        # Handle different input types for maximum compatibility
        if isinstance(batch, dict):
            src_tokens = batch['src_tokens']
            src_mask = batch.get('src_mask', None)
        elif isinstance(batch, tuple):
            src_tokens = batch[0]
            src_mask = None
        else:
            src_tokens = batch
            src_mask = None
        
        # Get batch size
        batch_size = src_tokens.size(0)
        device = src_tokens.device
        
        # Handle empty batch
        if batch_size == 0:
            return torch.zeros(0, 0, dtype=torch.long, device=device)
        
        # Create source mask if not provided
        if src_mask is None:
            src_mask = (src_tokens != self.pad_id)
        
        # Encode source
        encoder_outputs = self.forward({
            'src_tokens': src_tokens,
            'src_mask': src_mask
        })
        encoder_out = encoder_outputs['encoder_out']
        
        # Initialize target with BOS token
        tgt_tokens = torch.full(
            (batch_size, 1),
            self.bos_id,
            dtype=torch.long,
            device=device
        )
        
        # Generate tokens one by one
        for _ in range(max_len - 1):
            # Create target mask
            tgt_mask = torch.ones_like(tgt_tokens, dtype=torch.bool)
            
            # Forward pass
            outputs = self.forward({
                'src_tokens': src_tokens,
                'src_mask': src_mask,
                'tgt_tokens': tgt_tokens,
                'tgt_mask': tgt_mask
            })
            
            # Get logits for the last token
            logits = outputs['logits'][:, -1, :]
            
            # Get next token (greedy decoding)
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # Append next token to target
            tgt_tokens = torch.cat([tgt_tokens, next_token], dim=1)
            
            # Check if all sequences have EOS
            if (next_token == self.eos_id).all():
                break
        
        return tgt_tokens
