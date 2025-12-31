# models/boundary_predictor.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinnedBoundaryPredictor(nn.Module):
    """
    Two-stage boundary prediction:
    Stage 1: Coarse binning (6 bins) - easy classification
    Stage 2: Fine-grained offset regression within bin
    """
    
    def __init__(
        self,
        input_dim: int = 96,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Bin edges (6 bins covering [11, 128])
        self.register_buffer('bin_edges', torch.tensor([11, 30, 45, 60, 80, 100, 128]))
        self.num_bins = 6
        
        # Bin centers for initial prediction
        bin_centers = [(self.bin_edges[i] + self.bin_edges[i+1]) / 2 for i in range(self.num_bins)]
        self.register_buffer('bin_centers', torch.tensor(bin_centers))
        
        # Bin widths for offset normalization
        bin_widths = [self.bin_edges[i+1] - self.bin_edges[i] for i in range(self.num_bins)]
        self.register_buffer('bin_widths', torch.tensor(bin_widths))
        
        # Stage 1: Bin classifier (6-way classification)
        # Simplified to 2 layers: input_dim -> hidden_dim -> num_bins
        self.bin_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_bins)
        )
        
        # Stage 2: Offset regressor (continuous within bin)
        # Simplified to 2 layers: (input_dim + num_bins) -> hidden_dim -> 1
        self.offset_regressor = nn.Sequential(
            nn.Linear(input_dim + self.num_bins, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output in [-1, 1]
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
            event_refined: (B, N, input_dim)
        
        Returns:
            length_logits: (B, N, num_bins) - bin logits for compatibility
            length_dist: (B, N, num_bins) - bin probabilities
            predicted_length: (B, N) - continuous length prediction
        """
        B, N, _ = event_refined.shape
        
        # Stage 1: Predict bin
        bin_logits = self.bin_classifier(event_refined)  # (B, N, 6)
        bin_probs = F.softmax(bin_logits, dim=-1)
        bin_pred = torch.argmax(bin_probs, dim=-1)  # (B, N)
        
        # Stage 2: Predict offset within bin
        bin_onehot = F.one_hot(bin_pred, num_classes=self.num_bins).float()
        offset_input = torch.cat([event_refined, bin_onehot], dim=-1)
        offset_normalized = self.offset_regressor(offset_input).squeeze(-1)  # (B, N) in [-1, 1]
        
        # Combine: length = bin_center + offset * (bin_width / 2)
        bin_centers = self.bin_centers[bin_pred]  # (B, N)
        bin_widths = self.bin_widths[bin_pred]  # (B, N)
        
        predicted_length = bin_centers + offset_normalized * (bin_widths / 2.0)
        predicted_length = torch.clamp(predicted_length, 11, 128)
        
        return bin_logits, bin_probs, predicted_length
    
    def sample_length(self, length_dist, temperature=1.0, deterministic=False, 
                     soft_boundary=False, top_k=3):
        """
        Sample length from distribution
        
        Args:
            length_dist: (B, N, num_bins) - bin probabilities
            temperature: Sampling temperature
            deterministic: Use argmax
            soft_boundary: Use top-k sampling
            top_k: Number of top bins to sample from
        
        Returns:
            (B, N) - sampled bin indices (not actual lengths!)
        """
        if deterministic:
            return torch.argmax(length_dist, dim=-1)
        
        if soft_boundary:
            B, N, _ = length_dist.shape
            topk_probs, topk_indices = torch.topk(length_dist, k=min(top_k, self.num_bins), dim=-1)
            topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-10)
            
            logits = torch.log(topk_probs + 1e-10) / temperature
            probs = F.softmax(logits, dim=-1)
            
            probs_flat = probs.view(B * N, -1)
            sampled_topk_idx = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            sampled_topk_idx = sampled_topk_idx.view(B, N, 1)
            samples = torch.gather(topk_indices, -1, sampled_topk_idx).squeeze(-1)
            
            return samples
        else:
            logits = torch.log(length_dist + 1e-10) / temperature
            B, N, L = logits.shape
            
            logits_flat = logits.view(B * N, L)
            probs_flat = F.softmax(logits_flat, dim=-1)
            samples = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
            
            return samples.view(B, N)
    
    def compute_loss(self, bin_logits, predicted_length, true_length, mask=None):
        """
        Hybrid loss: bin classification + regression
        
        Args:
            bin_logits: (B, N, num_bins)
            predicted_length: (B, N) - continuous prediction
            true_length: (B, N) - ground truth
            mask: (B, N) - valid event mask
        
        Returns:
            total_loss: Scalar loss
        """
        B, N = true_length.shape
        
        # Get true bin for each length
        true_bins = torch.zeros_like(true_length, dtype=torch.long)
        for i in range(self.num_bins):
            if i == self.num_bins - 1:
                # Last bin: include upper bound
                bin_mask = (true_length >= self.bin_edges[i]) & (true_length <= self.bin_edges[i+1])
            else:
                bin_mask = (true_length >= self.bin_edges[i]) & (true_length < self.bin_edges[i+1])
            true_bins[bin_mask] = i
        
        # Bin classification loss
        bin_logits_flat = bin_logits.view(B * N, -1)
        true_bins_flat = true_bins.view(B * N)
        bin_loss = F.cross_entropy(bin_logits_flat, true_bins_flat, reduction='none')
        
        # Regression loss
        regression_loss = F.mse_loss(predicted_length, true_length.float(), reduction='none')
        
        # Combine losses
        total_loss = bin_loss + regression_loss
        
        if mask is not None:
            mask_flat = mask.view(B * N)
            total_loss = (total_loss * mask_flat).sum() / (mask_flat.sum() + 1e-8)
        else:
            total_loss = total_loss.mean()
        
        return total_loss


# Alias for backward compatibility
SimpleBoundaryPredictor = BinnedBoundaryPredictor
BoundaryDistributionPredictor = BinnedBoundaryPredictor