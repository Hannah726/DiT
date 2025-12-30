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
    
    Predicts: P(length | event_refined) for each event
    
    Args:
        input_dim: Input dimension (event_refined dimension, typically pattern_dim)
        hidden_dim: MLP hidden dimension
        max_len: Maximum sequence length
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 96,  # Changed from boundary_dim to input_dim (pattern_dim)
        hidden_dim: int = 128,
        max_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
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
    
    def forward(self, event_refined):
        """
        Args:
            event_refined: (B, N, input_dim) - refined event latents (from PatternDiscoveryPrompts)
        
        Returns:
            length_logits: (B, N, max_len+1) - raw logits
            length_dist: (B, N, max_len+1) - probability distribution
        """
        length_logits = self.predictor(event_refined)
        length_dist = F.softmax(length_logits, dim=-1)
        
        return length_logits, length_dist
    
    def sample_length(self, length_dist, temperature=1.0, deterministic=False, 
                     soft_boundary=False, top_k=3):
        """
        Sample or select sequence length with optional soft boundary
        
        Args:
            length_dist: (B, N, max_len+1) - probability distribution
            temperature: Sampling temperature (lower = more deterministic)
            deterministic: If True, use argmax
            soft_boundary: If True, use top-k sampling for robustness
                          This provides uncertainty and avoids hard truncation
            top_k: Number of top candidates to consider in soft boundary mode
        
        Returns:
            (B, N) - predicted lengths [0, max_len]
        """
        if deterministic:
            return torch.argmax(length_dist, dim=-1)
        
        if soft_boundary:
            # Soft boundary: sample from top-k candidates
            # This provides robustness against single-point failures
            # and increases length uncertainty for better generalization
            B, N, L = length_dist.shape
            
            # Get top-k candidates
            topk_probs, topk_indices = torch.topk(length_dist, k=min(top_k, L), dim=-1)
            # Renormalize top-k probabilities
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
            
            # Sample from top-k (with temperature)
            logits = torch.log(topk_probs + 1e-10) / temperature
            probs = F.softmax(logits, dim=-1)
            
            # Sample indices in top-k space
            probs_flat = probs.view(B * N, -1)
            sampled_topk_idx = torch.multinomial(
                probs_flat, 
                num_samples=1
            ).squeeze(-1)  # (B*N,)
            
            # Map back to original length space
            sampled_topk_idx = sampled_topk_idx.view(B, N, 1)
            samples = torch.gather(topk_indices, -1, sampled_topk_idx).squeeze(-1)
            
            return samples
        else:
            # Standard sampling from full distribution
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
