"""
EHR Diffusion Model with RQ-VAE Codes
Uses discrete codes instead of raw tokens
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from models.code_embedder import CodeEmbedder
from models.code_decoder import CodeDecoder
from models.dit import DiT


class EHRDiffusionCodesModel(nn.Module):
    """
    EHR Diffusion Model using RQ-VAE codes
    
    Architecture:
        1. CodeEmbedder: codes (8d discrete) -> code_latent (128d continuous)
        2. DiT: Denoising with time conditioning
        3. CodeDecoder: code_latent -> predicted codes -> (optionally) decode to tokens
    
    Args:
        codebook_size: RQ-VAE codebook size (default: 1024)
        rqvae_dim: RQ-VAE codebook dimension (default: 256)
        latent_dim: Diffusion latent dimension (default: 128)
        num_codes: Number of codes per event (default: 8)
        hidden_dim: DiT hidden dimension (default: 512)
        num_layers: DiT layers (default: 8)
        num_heads: Attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        freeze_codebook: Whether to freeze codebook (default: False)
        time_condition_dim: Time conditioning dimension (default: None, uses hidden_dim)
        use_sinusoidal_time: Use sinusoidal time encoding (default: True)
    """
    
    def __init__(
        self,
        codebook_size: int = 1024,
        rqvae_dim: int = 256,
        latent_dim: int = 128,
        num_codes: int = 8,
        hidden_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_codebook: bool = False,
        time_condition_dim: int = None,
        use_sinusoidal_time: bool = True
    ):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.num_codes = num_codes
        
        self.code_embedder = CodeEmbedder(
            codebook_size=codebook_size,
            rqvae_dim=rqvae_dim,
            latent_dim=latent_dim,
            num_codes=num_codes,
            aggregation='mean',
            freeze_codebook=freeze_codebook
        )
        
        self.dit = DiT(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            time_condition_dim=time_condition_dim,
            use_sinusoidal_time=use_sinusoidal_time
        )
        
        self.code_decoder = CodeDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            codebook_size=codebook_size,
            num_codes=num_codes,
            dropout=dropout
        )
    
    def load_rqvae_codebook(self, rqvae_checkpoint_path):
        """
        Load pretrained codebook from RQ-VAE
        
        Args:
            rqvae_checkpoint_path: Path to RQ-VAE checkpoint
        """
        self.code_embedder.load_rqvae_codebook(rqvae_checkpoint_path)
    
    def encode(self, codes):
        """
        Encode codes to continuous latent
        
        Args:
            codes: (B, N, 8) - discrete codes from RQ-VAE
        
        Returns:
            code_latent: (B, N, latent_dim) - continuous latent vectors
        """
        return self.code_embedder(codes)
    
    def decode(self, code_latent, return_logits=False):
        """
        Decode latent to codes
        
        Args:
            code_latent: (B, N, latent_dim)
            return_logits: Whether to return logits or sampled codes
        
        Returns:
            codes: (B, N, 8) or logits (B, N, 8, 1024)
        """
        return self.code_decoder(code_latent, return_logits=return_logits)
    
    def forward(self, codes):
        """
        Full forward pass (for validation)
        
        Args:
            codes: (B, N, 8) - discrete codes
        
        Returns:
            predicted_codes: (B, N, 8) - reconstructed codes
        """
        code_latent = self.encode(codes)
        predicted_codes = self.decode(code_latent, return_logits=False)
        return predicted_codes