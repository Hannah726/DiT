"""
Hybrid Time Encoder: Sinusoidal + MLP
"""

import torch
import torch.nn as nn
import math


class HybridTimeEncoder(nn.Module):
    """
    Hybrid Time Encoder combining sinusoidal and MLP encodings
    
    Input format (from dataset):
        - log1p(cumulative_minutes) from first event
        - Valid values: [0.0, ~7.88] (0 mins to 24 hours)
        - Padding: -1.0
        - First event is always 0.0
    
    Architecture:
        Branch A: Sinusoidal encoding
            - Multiple frequencies capture temporal patterns
            - Good for extrapolation and periodicity
        Branch B: MLP encoding
            - Learnable nonlinear transformation
            - Captures data-specific patterns
        Fusion: Concatenate both branches
    
    Args:
        input_dim: Input dimension (default: 1)
        time_dim: Output embedding dimension (must be even for concatenation)
        hidden_dim: Hidden dimension for MLP
        num_frequencies: Number of sinusoidal frequency bands
        max_time: Expected max time value (default: 8.0 for 24h window)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        time_dim: int = 32,
        hidden_dim: int = 128,
        num_frequencies: int = 16,
        max_time: float = 8.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert time_dim % 2 == 0, "time_dim must be even for sin+mlp concatenation"
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.num_frequencies = num_frequencies
        self.max_time = max_time
        
        # Sinusoidal frequencies: similar to transformer positional encoding
        # freq_k = 1 / (max_time^(2k/d)) for k in [0, num_frequencies)
        freq_bands = torch.pow(
            max_time,
            -torch.linspace(0, 1, num_frequencies, dtype=torch.float32)
        )
        self.register_buffer('freq_bands', freq_bands)  # (num_frequencies,)
        
        # Branch A: Project sinusoidal features
        sin_out_dim = time_dim // 2
        self.sin_proj = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_dim),  # *2 for sin and cos
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, sin_out_dim)
        )
        
        # Branch B: MLP encoding
        mlp_out_dim = time_dim // 2
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mlp_out_dim)
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(time_dim)
        
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
    
    def forward(self, log_time):
        """
        Args:
            log_time: (B, N, 1) - log1p(cumulative_minutes)
                      Valid: >= 0.0, Padding: -1.0
        
        Returns:
            (B, N, time_dim) - hybrid time embeddings
        """
        # Branch A: Sinusoidal encoding
        # log_time: (B, N, 1) -> angles: (B, N, num_frequencies)
        angles = log_time * self.freq_bands.unsqueeze(0).unsqueeze(0)
        
        sin_features = torch.sin(angles)  # (B, N, num_frequencies)
        cos_features = torch.cos(angles)  # (B, N, num_frequencies)
        
        sin_cos = torch.cat([sin_features, cos_features], dim=-1)  # (B, N, 2*num_frequencies)
        sin_emb = self.sin_proj(sin_cos)  # (B, N, time_dim//2)
        
        # Branch B: MLP encoding
        mlp_emb = self.mlp_branch(log_time)  # (B, N, time_dim//2)
        
        # Fusion: Concatenate
        time_emb = torch.cat([sin_emb, mlp_emb], dim=-1)  # (B, N, time_dim)
        
        # Output normalization
        time_emb = self.output_norm(time_emb)
        
        return time_emb


# Alias for backward compatibility
TimeEncoder = HybridTimeEncoder