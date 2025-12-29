"""
Diffusion Transformer (DiT) for EHR Latent Space
Event-level sequence modeling with timestep conditioning
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal positional embeddings for timestep encoding
    """
    
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
    """
    Embeds scalar timesteps into vector representations
    """
    
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


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization conditioned on timestep and demographics
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
            c: (B, C) - condition embedding
        Returns:
            (B, N, D) - modulated features
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = self.ln(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive normalization
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
            c: (B, C) - condition embedding
            mask: (B, N) - attention mask
        Returns:
            (B, N, D) - output sequence
        """
        # Self-attention with adaptive norm
        x_norm = self.norm1(x, c)
        
        # Convert mask for MultiheadAttention
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()  # (B, N)
        
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=attn_mask)
        x = x + attn_out
        
        # FFN with adaptive norm
        x = x + self.mlp(self.norm2(x, c))
        
        return x


class PromptCrossAttentionBlock(nn.Module):
    """
    DiT Block with Prompt Cross-Attention
    
    Architecture:
        1. Self-attention on latent events (events attend to events)
        2. Cross-attention to prompts (events attend to prompts) 
        3. Feed-forward network
    
    All with adaptive layer normalization conditioned on timestep + demographics
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
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm_cross = AdaptiveLayerNorm(hidden_dim, condition_dim)
        self.cross_attn = nn.MultiheadAttention(
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
    
    def forward(self, x, c, prompts=None, mask=None):
        """
        Args:
            x: (B, N, D) - latent event sequence
            c: (B, C) - condition embedding (timestep + demographics)
            prompts: (B, P, D) - prompt tokens from AdaptivePromptGenerator
            mask: (B, N) - valid event mask
        
        Returns:
            (B, N, D) - updated latent sequence
        """
        x_norm = self.norm1(x, c)
        
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask.bool()
        
        self_attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm, 
            key_padding_mask=attn_mask
        )
        x = x + self_attn_out
        
        if prompts is not None:
            x_norm_cross = self.norm_cross(x, c)
            
            cross_attn_out, _ = self.cross_attn(
                x_norm_cross,
                prompts,
                prompts
            )
            x = x + cross_attn_out
        
        x = x + self.mlp(self.norm2(x, c))
        
        return x


class DiT(nn.Module):
    """
    Diffusion Transformer for EHR Latent Denoising
    
    Args:
        latent_dim: Dimension of input latent (event_dim + time_dim)
        hidden_dim: Hidden dimension of transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        condition_dim: Dimension of conditioning (demographics)
        dropout: Dropout rate
        mlp_ratio: MLP hidden dimension ratio
        use_prompts: Whether to use prompt cross-attention
    """
    
    def __init__(
        self,
        latent_dim: int = 96,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        condition_dim: int = 2,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        use_prompts: bool = False
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_prompts = use_prompts
        
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        self.time_embedder = TimestepEmbedder(hidden_dim)
        
        self.condition_embedder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        combined_condition_dim = hidden_dim
        
        self.pos_embedding = nn.Parameter(torch.randn(1, 256, hidden_dim) * 0.02)
        
        if use_prompts:
            self.prompt_proj = nn.Linear(latent_dim, hidden_dim)
            self.blocks = nn.ModuleList([
                PromptCrossAttentionBlock(
                    hidden_dim,
                    num_heads,
                    combined_condition_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])
        else:
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
    
    def forward(self, x, t, condition=None, prompts=None, mask=None):
        """
        Args:
            x: (B, N, latent_dim) - noisy latent codes
            t: (B,) - timestep indices
            condition: (B, condition_dim) - demographics
            prompts: (B, P, latent_dim) - adaptive prompts
            mask: (B, N) - valid event mask
            
        Returns:
            (B, N, latent_dim) - predicted noise
        """
        B, N, _ = x.shape
        
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :N, :]
        
        t_emb = self.time_embedder(t)
        
        if condition is not None:
            c_emb = self.condition_embedder(condition)
            c = t_emb + c_emb
        else:
            c = t_emb
        
        if self.use_prompts and prompts is not None:
            prompts = self.prompt_proj(prompts)
        
        for block in self.blocks:
            if self.use_prompts:
                x = block(x, c, prompts=prompts, mask=mask)
            else:
                x = block(x, c, mask=mask)
        
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        return x