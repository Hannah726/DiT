"""
Structured Event Encoder
Encodes (token, type, dpe) → compact event latent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ChannelEmbedding(nn.Module):
    """
    Channel-specific embedding for token/type/dpe
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, L) - token indices
        Returns:
            (B, L, D) - embeddings
        """
        emb = self.embedding(x)
        emb = self.layer_norm(emb)
        return emb


class CrossChannelFusion(nn.Module):
    """
    Gated fusion of multiple channels
    """
    
    def __init__(self, channel_dims: dict, hidden_dim: int):
        """
        Args:
            channel_dims: dict of {channel_name: embed_dim}
            hidden_dim: output dimension
        """
        super().__init__()
        self.channel_names = list(channel_dims.keys())
        
        # Per-channel projections
        self.channel_projs = nn.ModuleDict({
            f'proj_{name}': nn.Linear(dim, hidden_dim)
            for name, dim in channel_dims.items()
        })
        
        # Gating mechanism
        self.gate_nets = nn.ModuleDict({
            f'gate_{name}': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            for name in channel_dims.keys()
        })
        
    def forward(self, channel_embs: dict):
        """
        Args:
            channel_embs: dict of {channel_name: (B, L, D_c)}
            
        Returns:
            (B, L, hidden_dim) - fused representation
        """
        # Project each channel
        projected = {}
        for name in self.channel_names:
            projected[name] = self.channel_projs[f'proj_{name}'](channel_embs[name])
        
        # Compute gates based on mean-pooled representations
        gates = {}
        for name in self.channel_names:
            pooled = projected[name].mean(dim=1, keepdim=True)  # (B, 1, D)
            gates[name] = self.gate_nets[f'gate_{name}'](pooled)  # (B, 1, D)
        
        # Weighted sum
        fused = sum(gates[name] * projected[name] for name in self.channel_names)
        
        return fused


class AttentionPooling(nn.Module):
    """
    Attention-based pooling from sequence to single vector
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.scale = hidden_dim ** -0.5
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, D) - sequence
            mask: (B, L) - attention mask
            
        Returns:
            (B, D) - pooled vector
        """
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        
        # Attention scores
        scores = torch.bmm(query, x.transpose(1, 2)) * self.scale  # (B, 1, L)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)  # (B, 1, L)
        pooled = torch.bmm(attn, x).squeeze(1)  # (B, D)
        
        return pooled


class StructuredEventEncoder(nn.Module):
    """
    Main encoder: (token, type, dpe) @ each event → event latent
    
    Architecture:
        1. Channel-specific embeddings
        2. Cross-channel gated fusion
        3. Attention pooling across tokens
        4. Final projection to event_dim
    """
    
    def __init__(
        self,
        vocab_sizes: dict,  # {'token': int, 'type': int, 'dpe': int}
        embed_dims: dict,   # {'token': int, 'type': int, 'dpe': int}
        hidden_dim: int = 256,
        event_dim: int = 32,
        dropout: float = 0.1,
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.event_dim = event_dim
        
        # Channel embeddings
        self.token_emb = ChannelEmbedding(vocab_sizes['token'], embed_dims['token'], padding_idx)
        self.type_emb = ChannelEmbedding(vocab_sizes['type'], embed_dims['type'], padding_idx)
        self.dpe_emb = ChannelEmbedding(vocab_sizes['dpe'], embed_dims['dpe'], padding_idx)
        
        # Cross-channel fusion
        self.fusion = CrossChannelFusion(embed_dims, hidden_dim)
        
        # Attention pooling
        self.pooling = AttentionPooling(hidden_dim)
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, event_dim),
            nn.LayerNorm(event_dim)
        )
        
    def forward(self, input_ids, type_ids, dpe_ids, event_mask=None):
        """
        Args:
            input_ids: (B, N, L) - token IDs per event
            type_ids: (B, N, L) - type IDs per event
            dpe_ids: (B, N, L) - dpe IDs per event
            event_mask: (B, N, L) - mask for valid tokens
            
        Returns:
            (B, N, event_dim) - event latents
        """
        B, N, L = input_ids.shape
        
        # Flatten batch and event dimensions
        input_flat = rearrange(input_ids, 'b n l -> (b n) l')
        type_flat = rearrange(type_ids, 'b n l -> (b n) l')
        dpe_flat = rearrange(dpe_ids, 'b n l -> (b n) l')
        
        # Embed each channel
        token_emb = self.token_emb(input_flat)  # (B*N, L, D_tok)
        type_emb = self.type_emb(type_flat)      # (B*N, L, D_typ)
        dpe_emb = self.dpe_emb(dpe_flat)         # (B*N, L, D_dpe)
        
        # Cross-channel fusion
        channel_embs = {
            'token': token_emb,
            'type': type_emb,
            'dpe': dpe_emb
        }
        fused = self.fusion(channel_embs)  # (B*N, L, hidden_dim)
        
        # Token-level mask (if provided)
        token_mask = None
        if event_mask is not None:
            token_mask = rearrange(event_mask, 'b n l -> (b n) l')
        
        # Pool across tokens
        pooled = self.pooling(fused, mask=token_mask)  # (B*N, hidden_dim)
        
        # Project to event dimension
        event_latents = self.proj(pooled)  # (B*N, event_dim)
        
        # Reshape back
        event_latents = rearrange(event_latents, '(b n) d -> b n d', b=B, n=N)
        
        return event_latents