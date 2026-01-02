"""
EHR Diffusion Model (Baseline Version)
Time-Conditional Generation: p(event | time)
Time is used as condition, not generated
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from models.event_encoder import StructuredEventEncoder
from models.dit import DiT
from models.event_decoder import EventDecoder


class EHRDiffusionModel(nn.Module):
    """
    EHR Diffusion Model (Baseline Version)
    Time-Conditional Generation: p(event | time)
    
    Architecture:
        1. StructuredEventEncoder: (token, type, dpe) -> event_latent (64)
        2. DiT: Denoising with time conditioning (time as condition, not generated)
        3. EventDecoder: Reconstruct events with validity-based boundary
    
    Args:
        vocab_size: Token vocabulary size
        type_vocab_size: Type vocabulary size
        dpe_vocab_size: DPE vocabulary size
        event_dim: Event latent dimension (default: 64)
        hidden_dim: DiT hidden dimension
        num_layers: DiT layers
        num_heads: Attention heads
        dropout: Dropout rate
        max_token_len: Max tokens per event
        time_condition_dim: Dimension for time conditioning (default: same as hidden_dim)
        use_sinusoidal_time: Whether to use sinusoidal encoding for time
    """
    
    def __init__(
        self,
        vocab_size: int = 2385,
        type_vocab_size: int = 7,
        dpe_vocab_size: int = 15,
        event_dim: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_token_len: int = 128,
        time_condition_dim: int = None,
        use_sinusoidal_time: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.event_dim = event_dim
        
        # Latent dimension is now only event (no time)
        self.latent_dim = event_dim
        
        self.event_encoder = StructuredEventEncoder(
            vocab_sizes={
                'token': vocab_size,
                'type': type_vocab_size,
                'dpe': dpe_vocab_size
            },
            embed_dims={
                'token': 128,
                'type': 32,
                'dpe': 32
            },
            hidden_dim=256,
            event_dim=event_dim,
            dropout=dropout
        )
        
        self.dit = DiT(
            latent_dim=self.latent_dim,  # 64 (only event)
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            time_condition_dim=time_condition_dim,
            use_sinusoidal_time=use_sinusoidal_time
        )
        
        self.event_decoder = EventDecoder(
            event_dim=event_dim,  # Changed from pattern_dim to event_dim
            hidden_dim=256,
            vocab_sizes={
                'token': vocab_size,
                'type': type_vocab_size,
                'dpe': dpe_vocab_size
            },
            max_token_len=max_token_len,
            dropout=dropout
        )
    
    def encode(self, input_ids, type_ids, dpe_ids, event_mask=None):
        """
        Encode structured events to latent representation
        
        Args:
            input_ids: (B, N, L) - token IDs
            type_ids: (B, N, L) - type IDs
            dpe_ids: (B, N, L) - DPE IDs
            event_mask: (B, N, L) - valid token mask
        
        Returns:
            event_latent: (B, N, event_dim) - event latent codes
            event_level_mask: (B, N) - valid event mask (1 for valid, 0 for padding)
        """
        event_latent = self.event_encoder(
            input_ids, type_ids, dpe_ids,
            event_mask=event_mask
        )
        
        # Compute event_level_mask
        if event_mask is not None:
            event_mask_sum = event_mask.sum(dim=-1)  # (B, N)
            event_level_mask = (event_mask_sum > 0).float()  # (B, N)
        else:
            # If no event_mask provided, assume all events are valid
            B, N = event_latent.shape[:2]
            event_level_mask = torch.ones(B, N, device=event_latent.device)
        
        return event_latent, event_level_mask
    
    def decode(self, event_latent, return_logits=False):
        """
        Decode event latent back to tokens
        
        Args:
            event_latent: (B, N, event_dim) - event latents
            return_logits: Whether to return logits
        
        Returns:
            dict with keys 'token', 'type', 'dpe' (and 'validity' if return_logits=True)
        """
        return self.event_decoder(
            event_latent,
            return_logits=return_logits
        )
    
    def forward(self, input_ids, type_ids, dpe_ids, event_mask=None):
        """
        Full forward pass (for debugging/validation, not used in training)
        
        Args:
            input_ids: (B, N, L) - token IDs
            type_ids: (B, N, L) - type IDs
            dpe_ids: (B, N, L) - DPE IDs
            event_mask: (B, N, L) - valid token mask
        
        Returns:
            dict with keys 'token', 'type', 'dpe'
        """
        if event_mask is None:
            event_mask = (input_ids > 0).float()
        
        event_latent, _ = self.encode(
            input_ids, type_ids, dpe_ids,
            event_mask=event_mask
        )
        
        decoded = self.decode(event_latent, return_logits=False)
        
        return {
            'token': decoded['token'],
            'type': decoded['type'],
            'dpe': decoded['dpe']
        }