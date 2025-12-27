"""
Time Decoder: Time Embedding -> Continuous Time
Decodes time embeddings back to continuous time intervals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeDecoder(nn.Module):
    """
    Decodes time embedding vectors back to continuous time intervals
    
    Architecture:
        MLP that reverses the TimeEncoder transformation
        Outputs continuous time values (log-normalized)
    
    Args:
        time_dim: Input time embedding dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (usually 1 for Delta-t)
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
        
        # Initialize weights
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
            (B, N, output_dim) - continuous time intervals (log-normalized)
        """
        con_time = self.mlp(time_emb)
        return con_time
    
    def compute_reconstruction_loss(self, time_emb, target_time, mask=None):
        """
        Compute MSE loss for time reconstruction
        
        Args:
            time_emb: (B, N, time_dim) - time embeddings
            target_time: (B, N, 1) - ground truth continuous time
            mask: (B, N) - valid event mask
            
        Returns:
            loss: Scalar reconstruction loss
            loss_dict: Dictionary of loss components
        """
        predicted_time = self.forward(time_emb)  # (B, N, 1)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
            loss = F.mse_loss(
                predicted_time * mask_expanded,
                target_time * mask_expanded,
                reduction='sum'
            ) / mask.sum()
        else:
            loss = F.mse_loss(predicted_time, target_time)
        
        loss_dict = {
            'time_recon': loss.item()
        }
        
        return loss, loss_dict

