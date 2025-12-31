"""
Complete EHR Diffusion Model with Pattern Discovery
Integrates: HybridTimeEncoder + PatternDiscoveryPrompts + BoundaryPredictor
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from models.event_encoder import StructuredEventEncoder
from models.time_encoder import HybridTimeEncoder
from models.demographic_encoder import DemographicEncoder
from models.pattern_prompts import PatternDiscoveryPrompts
from models.boundary_predictor import SimpleBoundaryPredictor
from models.dit import DiT
from models.event_decoder import EventDecoder
from models.time_decoder import TimeDecoder


class EHRDiffusionModel(nn.Module):
    """
    Complete EHR Joint Event-Time Diffusion Model with Pattern Discovery
    
    Architecture:
        1. StructuredEventEncoder: (token, type, dpe) -> event_latent (64)
        2. HybridTimeEncoder: log_time -> time_emb (32)
        3. PatternDiscoveryPrompts: (event, time) -> (event_refined, time_refined, prompts)
        4. SimpleBoundaryPredictor: event_refined -> length_dist (boundary NOT in diffusion)
        5. Joint Latent: [event_refined, time_refined] (192) - NO boundary
        6. DiT: Denoising with prompt conditioning (only event + time)
        7. Decoders: Reconstruct events, time, and boundary (from denoised event)
    
    Args:
        vocab_size: Token vocabulary size
        type_vocab_size: Type vocabulary size
        dpe_vocab_size: DPE vocabulary size
        event_dim: Event latent dimension
        time_dim: Time embedding dimension
        hidden_dim: DiT hidden dimension
        pattern_dim: PatternDiscoveryPrompts hidden dimension
        num_prompts: Number of learnable patterns
        num_layers: DiT layers
        num_heads: Attention heads
        demographic_dim: Demographics dimension
        dropout: Dropout rate
        max_token_len: Max tokens per event
        use_prompts: Whether to use pattern discovery
    """
    
    def __init__(
        self,
        vocab_size: int = 2385,
        type_vocab_size: int = 7,
        dpe_vocab_size: int = 15,
        event_dim: int = 64,
        time_dim: int = 32,
        hidden_dim: int = 512,
        pattern_dim: int = 96,
        num_prompts: int = 16,
        num_layers: int = 12,
        num_heads: int = 8,
        demographic_dim: int = 2,
        dropout: float = 0.1,
        max_token_len: int = 128,
        use_prompts: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.event_dim = event_dim
        self.time_dim = time_dim
        self.pattern_dim = pattern_dim
        self.use_prompts = use_prompts
        
        # Joint latent only contains event_refined + time_refined (no boundary)
        self.latent_dim = pattern_dim + pattern_dim  # 96 + 96 = 192
        
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
        
        self.time_encoder = HybridTimeEncoder(
            input_dim=1,
            time_dim=time_dim,
            hidden_dim=128,
            num_frequencies=16,
            max_time=8.0,
            dropout=dropout
        )
        
        self.pattern_prompts = PatternDiscoveryPrompts(
            event_dim=event_dim,
            time_dim=time_dim,
            hidden_dim=pattern_dim,
            num_prompts=num_prompts,
            num_heads=4,
            dropout=dropout
        )
        
        # Boundary predictor now takes event_refined as input (not boundary_emb)
        self.boundary_predictor = SimpleBoundaryPredictor(
            input_dim=pattern_dim,  # Takes event_refined (pattern_dim) as input
            hidden_dim=128,
            max_len=max_token_len,
            dropout=dropout
        )
        
        # DiT now only processes event_refined + time_refined (no boundary)
        self.dit = DiT(
            latent_dim=self.latent_dim,  # 192 (event + time only)
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            condition_dim=demographic_dim,
            dropout=dropout,
            use_prompts=use_prompts
        )
        
        self.event_decoder = EventDecoder(
            event_dim=pattern_dim,
            hidden_dim=256,
            vocab_sizes={
                'token': vocab_size,
                'type': type_vocab_size,
                'dpe': dpe_vocab_size
            },
            max_token_len=max_token_len,
            dropout=dropout
        )
        
        self.time_decoder = TimeDecoder(
            time_dim=pattern_dim,
            hidden_dim=128,
            output_dim=1,
            dropout=dropout
        )
    
    def encode(self, input_ids, type_ids, dpe_ids, con_time, event_mask=None):
        """
        Encode structured events and time with pattern discovery
        
        Args:
            input_ids: (B, N, L) - token IDs
            type_ids: (B, N, L) - type IDs
            dpe_ids: (B, N, L) - DPE IDs
            con_time: (B, N, 1) - log1p(cumulative_minutes), padding=-1.0
            event_mask: (B, N, L) - valid token mask
        
        Returns:
            joint_latent: (B, N, 192) - [event_refined, time_refined] (NO boundary)
            event_refined: (B, N, 96)
            time_refined: (B, N, 96)
            prompt_weights: (B, N, K)
            true_length: (B, N) - ground truth lengths
        """
        event_latent = self.event_encoder(
            input_ids, type_ids, dpe_ids,
            event_mask=event_mask
        )
        
        time_emb = self.time_encoder(con_time)
        
        if event_mask is not None:
            event_level_mask = (event_mask.sum(dim=-1) > 0).float()
        else:
            event_level_mask = None
        
        event_refined, time_refined, prompt_weights = self.pattern_prompts(
            event_latent, time_emb,
            mask=event_level_mask
        )
        
        if event_mask is not None:
            true_length = event_mask.sum(dim=-1).long()
        else:
            true_length = (input_ids > 0).sum(dim=-1).long()
        
        # Joint latent only contains event_refined + time_refined
        # Boundary is NOT included in diffusion process
        joint_latent = torch.cat([
            event_refined,
            time_refined
        ], dim=-1)
        
        return joint_latent, event_refined, time_refined, prompt_weights, true_length
    
    def decode(self, event_latent, boundary_mask=None, target_length=None, return_logits=False):
        """
        Decode event latent back to tokens
        
        Args:
            event_latent: (B, N, 96) - refined event latents
            boundary_mask: (B, N, L) - boundary constraint mask (optional)
            target_length: (B, N) - target length for length-aware decoding (optional)
            return_logits: Whether to return logits
        
        Returns:
            dict with keys 'token', 'type', 'dpe'
        """
        return self.event_decoder(
            event_latent,
            boundary_mask=boundary_mask,
            target_length=target_length,
            return_logits=return_logits
        )
    
    def decode_time(self, time_emb):
        """
        Decode time embedding back to log1p(cumulative_minutes)
        
        Args:
            time_emb: (B, N, 96) - refined time embeddings
        
        Returns:
            (B, N, 1) - log1p(cumulative_minutes)
        """
        return self.time_decoder(time_emb)
    
    def predict_boundary(self, event_refined):
        """
        Predict boundary (length) distribution from event_refined
        
        Args:
            event_refined: (B, N, pattern_dim) - refined event latents
        
        Returns:
            length_logits: (B, N, num_bins) - bin classification logits
            length_dist: (B, N, num_bins) - bin probabilities
        """
        bin_logits, bin_probs, _ = self.boundary_predictor(event_refined)
        return bin_logits, bin_probs
    
    def decode_joint_latent(self, joint_latent, return_logits=False, 
                           deterministic_boundary=True, soft_boundary=False, 
                           boundary_temperature=1.0, top_k=3):
        """
        Decode joint latent to events, time, and boundary
        
        Args:
            joint_latent: (B, N, 192) - denoised joint latent (event + time only)
            return_logits: Whether to return logits for events
            deterministic_boundary: Use argmax for boundary (vs sampling)
            soft_boundary: If True, use top-k sampling for robust boundary prediction
            boundary_temperature: Temperature for boundary sampling (lower = more deterministic)
            top_k: Number of top candidates in soft boundary mode
        
        Returns:
            decoded_events: dict with 'token', 'type', 'dpe'
            decoded_time: (B, N, 1)
            predicted_length: (B, N)
            boundary_mask: (B, N, L)
        """
        # Split joint latent
        event_latent = joint_latent[..., :self.pattern_dim]
        time_latent = joint_latent[..., self.pattern_dim:2*self.pattern_dim]
        
        # Predict boundary from denoised event_latent
        # BinnedBoundaryPredictor returns: (bin_logits, bin_probs, predicted_length)
        bin_logits, bin_probs, predicted_length = self.boundary_predictor(event_latent)
        
        # Round to nearest integer
        predicted_length = torch.round(predicted_length).long()
        predicted_length = torch.clamp(predicted_length, 11, 128)
        
        B, N = predicted_length.shape
        L = self.event_decoder.max_token_len
        positions = torch.arange(L, device=predicted_length.device).unsqueeze(0).unsqueeze(0)
        boundary_mask = (positions < predicted_length.unsqueeze(-1)).float()
        
        # Use length-aware decoding with predicted_length
        decoded_events = self.event_decoder(
            event_latent,
            boundary_mask=boundary_mask,
            target_length=predicted_length,
            return_logits=return_logits
        )
        
        decoded_time = self.time_decoder(time_latent)
        
        return decoded_events, decoded_time, predicted_length, boundary_mask
    
    def forward(self, input_ids, type_ids, dpe_ids, con_time, demographics=None, mask=None):
        """
        Full forward pass (for debugging/validation, not used in training)
        """
        token_mask = (input_ids > 0).float()
        
        joint_latent, event_refined, time_refined, prompt_weights, true_length = self.encode(
            input_ids, type_ids, dpe_ids, con_time,
            event_mask=token_mask
        )
        
        decoded_events, decoded_time, predicted_length, boundary_mask = self.decode_joint_latent(
            joint_latent,
            return_logits=False,
            deterministic_boundary=True,
            soft_boundary=False  # Use deterministic for forward pass
        )
        
        return {
            'token': decoded_events['token'],
            'type': decoded_events['type'],
            'dpe': decoded_events['dpe'],
            'time': decoded_time,
            'predicted_length': predicted_length,
            'true_length': true_length
        }