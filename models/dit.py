"""
Diffusion Transformer (DiT) for EHR Latent Space
Event-level sequence modeling with timestep + time conditioning (baseline version)
Time is used as condition, not generated
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        """
        Args:
            time: (B,) - timestep indices
        Returns:
            (B, dim) - time embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations"""
    
    def __init__(self, hidden_dim, frequency_embedding_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.frequency_embedding = SinusoidalPositionEmbeddings(frequency_embedding_dim)
        
    def forward(self, t):
        """
        Args:
            t: (B,) - timestep indices
        Returns:
            (B, hidden_dim) - timestep embeddings
        """
        t_freq = self.frequency_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class TimeConditioner(nn.Module):
    """
    Time Conditioner: Encodes time values (log1p(cumulative_minutes)) into conditioning embeddings
    
    Options:
    - Simple MLP: Direct projection
    - Sinusoidal: Similar to positional encoding (better for extrapolation)
    
    Args:
        time_dim: Input time dimension (default: 1 for scalar time)
        condition_dim: Output condition dimension (should match hidden_dim)
        use_sinusoidal: Whether to use sinusoidal encoding
        num_frequencies: Number of frequency bands for sinusoidal encoding
    """
    
    def __init__(
        self,
        time_dim: int = 1,
        condition_dim: int = 512,
        use_sinusoidal: bool = True,
        num_frequencies: int = 16,
        max_time: float = 8.0
    ):
        super().__init__()
        self.time_dim = time_dim
        self.condition_dim = condition_dim
        self.use_sinusoidal = use_sinusoidal
        
        if use_sinusoidal:
            # Sinusoidal encoding (similar to positional encoding)
            # Good for extrapolation and periodicity
            freq_bands = torch.pow(
                max_time,
                -torch.linspace(0, 1, num_frequencies, dtype=torch.float32)
            )
            self.register_buffer('freq_bands', freq_bands)
            
            # Project sinusoidal features to condition_dim
            sin_cos_dim = num_frequencies * 2  # sin + cos
            self.proj = nn.Sequential(
                nn.Linear(sin_cos_dim, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim)
            )
        else:
            # Simple MLP
            self.proj = nn.Sequential(
                nn.Linear(time_dim, condition_dim),
                nn.SiLU(),
                nn.Linear(condition_dim, condition_dim)
            )
    
    def forward(self, time_values):
        """
        Args:
            time_values: (B, N, 1) - log1p(cumulative_minutes), padding=-1.0
        
        Returns:
            time_cond: (B, N, condition_dim) - time conditioning embeddings
        """
        if self.use_sinusoidal:
            # Sinusoidal encoding
            # time_values: (B, N, 1)
            angles = time_values * self.freq_bands.unsqueeze(0).unsqueeze(0)  # (B, N, num_frequencies)
            sin_features = torch.sin(angles)
            cos_features = torch.cos(angles)
            sin_cos = torch.cat([sin_features, cos_features], dim=-1)  # (B, N, 2*num_frequencies)
            time_cond = self.proj(sin_cos)  # (B, N, condition_dim)
        else:
            # Simple MLP
            time_cond = self.proj(time_values)  # (B, N, condition_dim)
        
        return time_cond


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep and time
    
    Now supports:
    - Timestep embedding (diffusion timestep)
    - Time conditioning (event time values)
    """
    
    def __init__(self, hidden_dim, condition_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * hidden_dim, bias=True)
        )
        
    def forward(self, x, c):
        """
        Args:
            x: (B, N, D) - input features
            c: (B, C) - combined condition embedding (timestep + time)
        Returns:
            (B, N, D) - modulated features
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.ln(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class DiTBlock(nn.Module):
    """
    Standard DiT block with self-attention
    Now uses combined conditioning (timestep + time)
    """
    
    def __init__(
        self,
        hidden_dim,
        num_heads,
        condition_dim,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        self.norm1 = AdaptiveLayerNorm(hidden_dim, condition_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = AdaptiveLayerNorm(hidden_dim, condition_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, c, mask=None):
        """
        Args:
            x: (B, N, D) - input sequence
            c: (B, C) - combined condition embedding (timestep + time pooled)
            mask: (B, N) - attention mask
        Returns:
            (B, N, D) - output sequence
        """
        x_norm = self.norm1(x, c)
        
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x, c))
        
        return x


class PromptCrossAttentionBlock(nn.Module):
    """
    DiT Block with Prompt Cross-Attention
    
    Architecture:
        1. Self-attention on latent events
        2. Cross-attention to prompts (NEW!)
        3. Feed-forward network
    
    All with adaptive layer normalization
    """
    
    def __init__(
        self,
        hidden_dim,
        num_heads,
        condition_dim,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = AdaptiveLayerNorm(hidden_dim, condition_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention to prompts
        self.norm_cross = AdaptiveLayerNorm(hidden_dim, condition_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward
        self.norm2 = AdaptiveLayerNorm(hidden_dim, condition_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, c, prompts=None, mask=None):
        """
        Args:
            x: (B, N, D) - latent event sequence
            c: (B, C) - condition embedding (timestep + demographics)
            prompts: (B, P, D) - prompt vectors (from PatternDiscoveryPrompts)
            mask: (B, N) - valid event mask
        
        Returns:
            (B, N, D) - updated latent sequence
        """
        # Self-attention
        x_norm = self.norm1(x, c)
        
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()
        
        self_attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=attn_mask
        )
        x = x + self_attn_out
        
        # Cross-attention to prompts
        if prompts is not None:
            x_norm_cross = self.norm_cross(x, c)
            
            cross_attn_out, _ = self.cross_attn(
                x_norm_cross,  # Query: latent events
                prompts,       # Key: prompts
                prompts        # Value: prompts
            )
            x = x + cross_attn_out
        
        # Feed-forward
        x = x + self.mlp(self.norm2(x, c))
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for EHR Latent Denoising (Baseline Version)
    
    Architecture:
        - Time-conditional generation: p(event | time)
        - Time is used as condition, not generated
        - Only event latent (64d) goes through diffusion
        - Time conditioning via AdaLN modulation
    
    Args:
        latent_dim: Dimension of event latent (default: 64, only event, no time)
        hidden_dim: Hidden dimension of transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        mlp_ratio: MLP hidden dimension ratio
        time_condition_dim: Dimension for time conditioning (default: same as hidden_dim)
        use_sinusoidal_time: Whether to use sinusoidal encoding for time
    """
    
    def __init__(
        self,
        latent_dim: int = 64,  # Only event latent, no time
        hidden_dim: int = 512,
        num_layers: int = 8,  # Can reduce from 12
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        time_condition_dim: int = None,
        use_sinusoidal_time: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        if time_condition_dim is None:
            time_condition_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Timestep embedder (for diffusion timestep)
        self.timestep_embedder = TimestepEmbedder(hidden_dim)
        
        # Time conditioner (for event time values)
        self.time_conditioner = TimeConditioner(
            time_dim=1,
            condition_dim=time_condition_dim,
            use_sinusoidal=use_sinusoidal_time
        )
        
        # Project time condition to hidden_dim if dimensions don't match
        if time_condition_dim != hidden_dim:
            self.time_cond_proj = nn.Linear(time_condition_dim, hidden_dim)
        else:
            self.time_cond_proj = None
        
        # Combined condition dimension (timestep + time)
        combined_condition_dim = hidden_dim
        
        # Positional embedding for events
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)
        
        # Transformer blocks (no prompts, pure self-attention)
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim,
                num_heads,
                combined_condition_dim,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # Initialize weights
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
    
    def forward(self, x, t, time_condition=None, mask=None):
        """
        Args:
            x: (B, N, latent_dim) - noisy event latent codes (64d)
            t: (B,) - diffusion timestep indices
            time_condition: (B, N, 1) - time values (log1p(cumulative_minutes))
                           If None, will use null condition (time=0)
            mask: (B, N) - valid event mask
            
        Returns:
            (B, N, latent_dim) - predicted noise
        """
        B, N, _ = x.shape
        
        # Project input
        x = self.input_proj(x)  # (B, N, hidden_dim)
        x = x + self.pos_embedding[:, :N, :]
        
        # Embed diffusion timestep
        timestep_emb = self.timestep_embedder(t)  # (B, hidden_dim)
        
        # Process time condition
        if time_condition is not None:
            # time_condition: (B, N, 1) - log1p(cumulative_minutes)
            time_cond = self.time_conditioner(time_condition)  # (B, N, time_condition_dim)
            
            # Pool time condition to (B, hidden_dim) for AdaLN
            # Option 1: Mean pooling (simple)
            if mask is not None:
                # Mask out padding positions
                mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
                time_cond_pooled = (time_cond * mask_expanded).sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # (B, time_condition_dim)
            else:
                time_cond_pooled = time_cond.mean(dim=1)  # (B, time_condition_dim)
            
            # Project to hidden_dim if needed
            if self.time_cond_proj is not None:
                time_cond_pooled = self.time_cond_proj(time_cond_pooled)
            
            # Combine timestep and time condition
            combined_cond = timestep_emb + time_cond_pooled  # (B, hidden_dim)
        else:
            # Null condition: only use timestep
            combined_cond = timestep_emb  # (B, hidden_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, combined_cond, mask=mask)
        
        # Output projection
        x = self.final_norm(x)
        x = self.output_proj(x)  # (B, N, latent_dim)
        
        return x