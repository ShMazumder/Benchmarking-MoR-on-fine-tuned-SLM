"""
Baseline Transformer Model
Standard fixed-depth Transformer for language modeling
"""
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=12, 
                 d_ff=2048, dropout=0.1, max_seq_len=64):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        """
        Args:
            x: Input token indices [batch, seq_len]
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            effective_depth: Always n_layers for baseline
        """
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        seq_len = x.size(1)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Pass through transformer
        x = self.transformer(x, mask=mask, is_causal=True)
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        # Effective depth is always n_layers for baseline
        effective_depth = torch.tensor(self.n_layers, dtype=torch.float32)
        
        return logits, effective_depth
