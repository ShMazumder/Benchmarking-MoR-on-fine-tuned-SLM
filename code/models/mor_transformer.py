"""
MoR Transformer Model
Transformer with Mixture-of-Recursion routing
"""
import torch
import torch.nn as nn
import math
from .router import MoRRouter

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

class MoRTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=12, 
                 d_ff=2048, dropout=0.1, max_seq_len=64, router_hidden_dim=128,
                 gumbel_temperature=1.0, **kwargs):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of shared recursive layers
            d_ff: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            router_hidden_dim: Router MLP hidden dimension
            gumbel_temperature: Temperature for Gumbel-Softmax
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Shared transformer layer (parameter sharing)
        self.shared_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        
        # Router for each layer
        self.routers = nn.ModuleList([
            MoRRouter(d_model, router_hidden_dim, gumbel_temperature)
            for _ in range(n_layers)
        ])
        
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
    
    def forward(self, x, training=True):
        """
        Args:
            x: Input token indices [batch, seq_len]
            training: If True, use Gumbel-Softmax; else use argmax
        
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            effective_depth: Average depth used [scalar]
            routing_stats: Dictionary with routing statistics
        """
        batch_size, seq_len = x.shape
        
        # Embed and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Track depth for each token
        token_depths = torch.zeros(batch_size, seq_len, device=x.device)
        
        # Routing statistics
        skip_count = 0
        forward_count = 0
        recurse_count = 0
        
        # Pass through layers with routing
        for layer_idx in range(self.n_layers):
            # Get routing decision
            actions, logits = self.routers[layer_idx](x, training=training)
            
            # Count actions
            skip_mask = (actions == 0)
            forward_mask = (actions == 1)
            recurse_mask = (actions == 2)
            
            skip_count += skip_mask.sum().item()
            forward_count += forward_mask.sum().item()
            recurse_count += recurse_mask.sum().item()
            
            # Apply layer based on routing
            # Skip: do nothing (cost 0)
            # Forward: apply layer once (cost 1)
            # Recurse: apply layer and increment depth (cost 1+)
            
            # For simplicity, we apply the layer to all tokens
            # and then mask the output based on routing
            x_transformed = self.shared_layer(x, src_mask=mask, is_causal=True)
            
            # Apply routing: skip keeps original, forward/recurse use transformed
            x = torch.where(
                skip_mask.unsqueeze(-1).expand_as(x),
                x,  # Skip: keep original
                x_transformed  # Forward/Recurse: use transformed
            )
            
            # Update depths
            token_depths += (~skip_mask).float()  # Add 1 for non-skip actions
        
        # Project to vocabulary
        logits = self.fc_out(x)
        
        # Calculate effective depth (average across all tokens)
        effective_depth = token_depths.mean()
        
        # Routing statistics
        total_decisions = batch_size * seq_len * self.n_layers
        routing_stats = {
            'skip_rate': skip_count / total_decisions,
            'forward_rate': forward_count / total_decisions,
            'recurse_rate': recurse_count / total_decisions,
            'effective_depth': effective_depth.item()
        }
        
        return logits, effective_depth, routing_stats
