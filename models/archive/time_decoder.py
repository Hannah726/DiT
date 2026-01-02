"""
Time Decoder: Time Embedding -> Continuous Time
Decodes time embeddings back to log1p(cumulative_minutes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDecoder(nn.Module):
    """
    Decodes time embeddings back to continuous time
    
    Output: log1p(cumulative_minutes), same format as input
    Note: No denormalization here - that's done at evaluation time
    
    Args:
        time_dim: Time embedding dimension (from HybridTimeEncoder)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (default: 1 for scalar time)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        time_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.time_dim = time_dim
        self.output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, time_emb):
        """
        Args:
            time_emb: (B, N, time_dim) - time embeddings
            
        Returns:
            (B, N, 1) - log1p(cumulative_minutes)
        """
        return self.mlp(time_emb)
    
    def compute_reconstruction_loss(self, time_emb, target_time, mask=None):
        """
        MSE loss for time reconstruction
        
        Args:
            time_emb: (B, N, time_dim)
            target_time: (B, N, 1) - log1p(cumulative_minutes)
            mask: (B, N) - valid event mask
        """
        predicted_time = self.forward(time_emb)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            loss = F.mse_loss(
                predicted_time * mask_expanded,
                target_time * mask_expanded,
                reduction='sum'
            ) / (mask.sum() + 1e-8)
        else:
            loss = F.mse_loss(predicted_time, target_time)
        
        return loss, {'time_recon': loss.item()}