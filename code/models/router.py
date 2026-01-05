"""
MoR Router Module
Lightweight MLP that decides: Skip, Forward, or Recurse
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoRRouter(nn.Module):
    def __init__(self, d_model=256, hidden_dim=128, temperature=1.0):
        """
        Args:
            d_model: Input dimension (token embedding size)
            hidden_dim: Hidden layer dimension
            temperature: Gumbel-Softmax temperature
        """
        super().__init__()
        self.temperature = temperature
        
        # MLP: d_model -> hidden_dim -> 3
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3 actions: Skip, Forward, Recurse
        )
    
    def forward(self, x, training=True):
        """
        Args:
            x: Token embeddings [batch, seq_len, d_model]
            training: If True, use Gumbel-Softmax; else use argmax
        
        Returns:
            actions: Routing decisions [batch, seq_len] (0=Skip, 1=Forward, 2=Recurse)
            logits: Raw logits [batch, seq_len, 3]
        """
        # Get logits for each token
        logits = self.mlp(x)  # [batch, seq_len, 3]
        
        if training:
            # Gumbel-Softmax for differentiable sampling
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            actions = torch.argmax(probs, dim=-1)  # [batch, seq_len]
        else:
            # Argmax for inference
            actions = torch.argmax(logits, dim=-1)  # [batch, seq_len]
        
        return actions, logits
    
    def get_action_distribution(self, x):
        """Get probability distribution over actions"""
        logits = self.mlp(x)
        return F.softmax(logits, dim=-1)
