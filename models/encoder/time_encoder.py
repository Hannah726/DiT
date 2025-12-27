"""
Time Encoder for continuous time intervals
Simple MLP with log-transformation
"""

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Encodes continuous time intervals (already log-normalized)
    
    Args:
        input_dim: Input dimension (usually 1 for Delta-t)
        time_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        time_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_dim = time_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, time_dim),
            nn.LayerNorm(time_dim)
        )
        
    def forward(self, con_time):
        """
        Args:
            con_time: (B, N, 1) - continuous time intervals (log-normalized)
            
        Returns:
            (B, N, time_dim) - time embeddings
        """
        time_emb = self.mlp(con_time)
        return time_emb
