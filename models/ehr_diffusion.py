"""
Complete EHR Diffusion Model
Combines all components: encoder, time encoder, demographic encoder, DiT, decoder
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from models.encoder.event_encoder import StructuredEventEncoder
from models.encoder.time_encoder import TimeEncoder
from models.conditioning.demographic_encoder import DemographicEncoder
from models.diffusion.dit import DiT
from models.decoder.event_decoder import EventDecoder
from models.decoder.time_decoder import TimeDecoder


class EHRDiffusionModel(nn.Module):
    """
    Complete EHR Joint Event-Time Diffusion Model
    
    Architecture:
        1. Structured Event Encoder: (token, type, dpe) -> event_latent
        2. Time Encoder: continuous_time -> time_emb
        3. Joint Latent: [event_latent, time_emb]
        4. Demographic Encoder: demographics -> condition
        5. DiT: Denoising diffusion in joint latent space
        6. Event Decoder: event_latent -> (token, type, dpe)
    
    Args:
        vocab_size: Size of token vocabulary
        type_vocab_size: Size of type vocabulary
        dpe_vocab_size: Size of DPE vocabulary
        event_dim: Event latent dimension
        time_dim: Time embedding dimension
        hidden_dim: DiT hidden dimension
        num_layers: Number of DiT layers
        num_heads: Number of attention heads
        demographic_dim: Input demographic dimension
        dropout: Dropout rate
        max_token_len: Maximum tokens per event
    """
    
    def __init__(
        self,
        vocab_size: int = 2385,
        type_vocab_size: int = 7,
        dpe_vocab_size: int = 15,
        event_dim: int = 64,
        time_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        demographic_dim: int = 2,
        dropout: float = 0.1,
        max_token_len: int = 128
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.event_dim = event_dim
        self.time_dim = time_dim
        self.latent_dim = event_dim + time_dim
        
        # 1. Structured Event Encoder
        self.encoder = StructuredEventEncoder(
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
        
        # 2. Time Encoder
        self.time_encoder = TimeEncoder(
            input_dim=1,
            time_dim=time_dim,
            hidden_dim=128,
            dropout=dropout
        )
        
        # 3. Demographic Encoder
        self.demographic_encoder = DemographicEncoder(
            demographic_dim=demographic_dim,
            hidden_dim=128,
            output_dim=256,
            dropout=dropout,
            use_embedding=False
        )
        
        # 4. Diffusion Transformer (DiT)
        self.dit = DiT(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            condition_dim=demographic_dim,
            dropout=dropout
        )
        
        # 5. Event Decoder
        self.decoder = EventDecoder(
            event_dim=event_dim,
            hidden_dim=256,
            vocab_sizes={
                'token': vocab_size,
                'type': type_vocab_size,
                'dpe': dpe_vocab_size
            },
            max_token_len=max_token_len,
            dropout=dropout
        )
        
        # 6. Time Decoder
        self.time_decoder = TimeDecoder(
            time_dim=time_dim,
            hidden_dim=128,
            output_dim=1,
            dropout=dropout
        )
    
    def encode(self, input_ids, type_ids, dpe_ids, con_time, event_mask=None):
        """
        Encode structured events and time into joint latent space
        
        Args:
            input_ids: (B, N, L) - token IDs
            type_ids: (B, N, L) - type IDs
            dpe_ids: (B, N, L) - DPE IDs
            con_time: (B, N, 1) - continuous time intervals
            event_mask: (B, N, L) - valid token mask
        
        Returns:
            joint_latent: (B, N, latent_dim) - joint event-time latent
            event_latent: (B, N, event_dim) - event latent only
            time_emb: (B, N, time_dim) - time embedding
        """
        # Encode events
        event_latent = self.encoder(input_ids, type_ids, dpe_ids, event_mask=event_mask)
        
        # Encode time
        time_emb = self.time_encoder(con_time)
        
        # Combine into joint latent
        joint_latent = torch.cat([event_latent, time_emb], dim=-1)
        
        return joint_latent, event_latent, time_emb
    
    def decode(self, event_latent, return_logits=False):
        """
        Decode event latent back to tokens
        
        Args:
            event_latent: (B, N, event_dim) - event latent codes
            return_logits: Whether to return logits or sampled IDs
        
        Returns:
            dict with keys 'token', 'type', 'dpe'
        """
        return self.decoder(event_latent, return_logits=return_logits)
    
    def decode_time(self, time_emb):
        """
        Decode time embedding back to continuous time
        
        Args:
            time_emb: (B, N, time_dim) - time embeddings
        
        Returns:
            (B, N, 1) - continuous time intervals (log-normalized)
        """
        return self.time_decoder(time_emb)
    
    def forward(
        self,
        input_ids,
        type_ids,
        dpe_ids,
        con_time,
        demographics,
        mask=None,
        return_latents=False
    ):
        """
        Full forward pass for reconstruction (debugging/validation)
        
        Args:
            input_ids: (B, N, L) - token IDs
            type_ids: (B, N, L) - type IDs
            dpe_ids: (B, N, L) - DPE IDs
            con_time: (B, N, 1) - continuous time intervals
            demographics: (B, D) - demographic features
            mask: (B, N) - valid event mask
            return_latents: Whether to return latent codes
        
        Returns:
            reconstructed: dict with keys 'token', 'type', 'dpe'
            (optional) latents: dict with latent representations
        """
        # Create token-level mask
        token_mask = (input_ids.sum(dim=-1) > 0).float()  # (B, N)
        token_mask_expanded = token_mask.unsqueeze(-1).expand_as(input_ids)  # (B, N, L)
        
        # Encode
        joint_latent, event_latent, time_emb = self.encode(
            input_ids, type_ids, dpe_ids, con_time, event_mask=token_mask_expanded
        )
        
        # Decode
        reconstructed = self.decode(event_latent, return_logits=False)
        reconstructed_time = self.decode_time(time_emb)
        reconstructed['time'] = reconstructed_time
        
        if return_latents:
            latents = {
                'joint_latent': joint_latent,
                'event_latent': event_latent,
                'time_emb': time_emb
            }
            return reconstructed, latents
        
        return reconstructed
    
    def compute_reconstruction_loss(
        self,
        input_ids,
        type_ids,
        dpe_ids,
        con_time,
        mask=None
    ):
        """
        Compute reconstruction loss for encoder-decoder
        
        Args:
            input_ids: (B, N, L) - token IDs
            type_ids: (B, N, L) - type IDs
            dpe_ids: (B, N, L) - DPE IDs
            con_time: (B, N, 1) - continuous time intervals
            mask: (B, N, L) - valid token mask
        
        Returns:
            loss: Scalar reconstruction loss
            loss_dict: Dictionary of loss components
        """
        # Encode
        _, event_latent, _ = self.encode(input_ids, type_ids, dpe_ids, con_time)
        
        # Compute reconstruction loss
        loss, loss_dict = self.decoder.compute_reconstruction_loss(
            event_latent,
            input_ids,
            type_ids,
            dpe_ids,
            mask=mask
        )
        
        return loss, loss_dict