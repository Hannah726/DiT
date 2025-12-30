"""
Simple Boundary Predictor
Predicts sequence length directly from boundary latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBoundaryPredictor(nn.Module):
    """
    Simple Boundary (Length) Predictor
    
    Predicts: P(length | boundary_latent) for each event
    
    Args:
        boundary_dim: Boundary latent dimension (from joint latent)
        hidden_dim: MLP hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        boundary_dim: int = 16,
        hidden_dim: int = 64,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.boundary_dim = boundary_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.predictor = nn.Sequential(
            nn.Linear(boundary_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_len + 1)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, boundary_latent):
        """
        Args:
            boundary_latent: (B, N, boundary_dim) - boundary latent from denoised
        
        Returns:
            length_logits: (B, N, max_len+1) - raw logits
            length_dist: (B, N, max_len+1) - probability distribution
        """
        length_logits = self.predictor(boundary_latent)
        length_dist = F.softmax(length_logits, dim=-1)
        
        return length_logits, length_dist
    
    def sample_length(self, length_dist, temperature=1.0, deterministic=False):
        """
        Sample or select sequence length
        
        Args:
            length_dist: (B, N, max_len+1) - probability distribution
            temperature: Sampling temperature (lower = more deterministic)
            deterministic: If True, use argmax
        
        Returns:
            (B, N) - predicted lengths [0, max_len]
        """
        if deterministic:
            return torch.argmax(length_dist, dim=-1)
        else:
            logits = torch.log(length_dist + 1e-10) / temperature
            B, N, L = logits.shape
            
            logits_flat = logits.view(B * N, L)
            probs_flat = F.softmax(logits_flat, dim=-1)
            samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            
            return samples.view(B, N)
    
    def compute_loss(self, length_logits, true_length, mask=None):
        """
        Cross-entropy loss for length prediction
        
        Args:
            length_logits: (B, N, max_len+1) - predicted logits
            true_length: (B, N) - ground truth lengths [0, max_len]
            mask: (B, N) - valid event mask
        
        Returns:
            loss: Scalar loss
        """
        B, N = true_length.shape
        
        logits_flat = length_logits.view(B * N, -1)
        targets_flat = true_length.view(B * N)
        
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        
        if mask is not None:
            mask_flat = mask.view(B * N)
            loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            loss = loss.mean()
        
        return loss


BoundaryDistributionPredictor = SimpleBoundaryPredictor
