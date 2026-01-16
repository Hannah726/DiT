# models/dit.py (新增时间embedding部分)

import torch
import torch.nn import nn
import math


class DiT(nn.Module):
    """
    Diffusion Transformer for EHR Code Generation
    
    Args:
        latent_dim: Dimension of code latent vectors
        hidden_dim: Hidden dimension for transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        time_condition_dim: Dimension for time condition embedding
        max_time_vocab: Maximum time vocabulary size
        time_token_len: Length of time token sequence per event
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        time_condition_dim: int = None,
        max_time_vocab: int = 100,
        time_token_len: int = 2,
        use_sinusoidal_time: bool = True
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.time_condition_dim = time_condition_dim or hidden_dim
        self.time_token_len = time_token_len
        
        # Time token embedding
        self.time_embedding = nn.Embedding(max_time_vocab, self.time_condition_dim)
        
        # Project time tokens to condition
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_condition_dim * time_token_len, self.time_condition_dim),
            nn.LayerNorm(self.time_condition_dim),
            nn.GELU()
        )
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Diffusion timestep embedding (sinusoidal)
        if use_sinusoidal_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
        
        # Transformer blocks with time conditioning
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                condition_dim=self.time_condition_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x, t, time_ids, mask=None):
        """
        Args:
            x: (B, N, latent_dim) - noisy code latents
            t: (B,) - diffusion timesteps
            time_ids: (B, N, time_token_len) - discrete time tokens
            mask: (B, N) - valid event mask
        
        Returns:
            (B, N, latent_dim) - denoised code latents
        """
        B, N, _ = x.shape
        
        # Encode time tokens
        time_emb = self.time_embedding(time_ids)  # (B, N, time_token_len, time_condition_dim)
        time_emb = time_emb.reshape(B, N, -1)  # (B, N, time_token_len * time_condition_dim)
        time_condition = self.time_proj(time_emb)  # (B, N, time_condition_dim)
        
        # Encode diffusion timestep
        t_emb = self.timestep_embedding(t, self.hidden_dim)  # (B, hidden_dim)
        t_emb = self.time_mlp(t_emb)  # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)
        
        # Project input
        h = self.input_proj(x)  # (B, N, hidden_dim)
        h = h + t_emb  # Add diffusion timestep
        
        # Apply transformer blocks with time conditioning
        for block in self.blocks:
            h = block(h, time_condition, mask)
        
        # Output projection
        out = self.output_proj(h)
        
        return out
    
    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        Sinusoidal timestep embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class DiTBlock(nn.Module):
    """
    Transformer block with time conditioning via cross-attention
    """
    
    def __init__(self, hidden_dim, num_heads, condition_dim, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True, 
            kdim=condition_dim, vdim=condition_dim
        )
        
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, time_condition, mask=None):
        """
        Args:
            x: (B, N, hidden_dim)
            time_condition: (B, N, condition_dim)
            mask: (B, N) - attention mask
        """
        # Self-attention
        attn_mask = ~mask.bool() if mask is not None else None
        x = x + self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=attn_mask
        )[0]
        
        # Cross-attention with time condition
        x = x + self.cross_attn(
            self.norm2(x), time_condition, time_condition,
            key_padding_mask=attn_mask
        )[0]
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        
        return x