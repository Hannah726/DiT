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
from models.boundary_predictor import BoundaryDistributionPredictor
from models.dit import DiT
from models.event_decoder import EventDecoder
from models.time_decoder import TimeDecoder


class EHRDiffusionModel(nn.Module):
    """
    Complete EHR Joint Event-Time Diffusion Model with Pattern Discovery
    
    NEW Architecture:
        1. StructuredEventEncoder: (token, type, dpe) -> event_latent (64)
        2. HybridTimeEncoder: log_time -> time_emb (32)
        3. PatternDiscoveryPrompts: (event, time) -> (event_refined, time_refined, prompts)
           - event_refined: 96, time_refined: 96
        4. BoundaryPredictor: (event_refined, prompts) -> length_dist
        5. Joint Latent: [event_refined, time_refined, boundary_emb] (208)
        6. DiT: Denoising with prompt conditioning
        7. Decoders: Reconstruct events, time, and boundary
    
    Args:
        vocab_size: Token vocabulary size
        type_vocab_size: Type vocabulary size
        dpe_vocab_size: DPE vocabulary size
        event_dim: Event latent dimension (from StructuredEventEncoder)
        time_dim: Time embedding dimension (from HybridTimeEncoder)
        hidden_dim: DiT hidden dimension
        pattern_dim: PatternDiscoveryPrompts hidden dimension
        num_prompts: Number of learnable patterns
        num_layers: DiT layers
        num_heads: Attention heads
        demographic_dim: Demographics dimension
        dropout: Dropout rate
        max_token_len: Max tokens per event
        use_prompts: Whether to use pattern discovery (should be True)
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
        
        # Dimensions
        self.boundary_dim = 16
        self.latent_dim = pattern_dim + pattern_dim + self.boundary_dim  # 96+96+16=208
        
        # 1. Structured Event Encoder
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
        
        # 2. Hybrid Time Encoder
        self.time_encoder = HybridTimeEncoder(
            input_dim=1,
            time_dim=time_dim,
            hidden_dim=128,
            num_frequencies=16,
            max_time=8.0,  # log1p(24*60) ≈ 7.88
            dropout=dropout
        )
        
        # 3. Pattern Discovery Prompts (Core Innovation!)
        self.pattern_prompts = PatternDiscoveryPrompts(
            event_dim=event_dim,
            time_dim=time_dim,
            hidden_dim=pattern_dim,
            num_prompts=num_prompts,
            num_heads=4,
            dropout=dropout
        )
        
        # 4. Boundary Predictor
        self.boundary_predictor = BoundaryDistributionPredictor(
            event_dim=pattern_dim,
            prompt_dim=pattern_dim,
            hidden_dim=128,
            max_len=max_token_len,
            dropout=dropout
        )
        
        # Boundary embedding (positional encoding for predicted length)
        self.boundary_embedding = nn.Embedding(max_token_len + 1, self.boundary_dim)
        
        # 5. Diffusion Transformer (DiT)
        self.dit = DiT(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            condition_dim=demographic_dim,
            dropout=dropout,
            use_prompts=use_prompts
        )
        
        # 6. Event Decoder
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
        
        # 7. Time Decoder
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
            joint_latent: (B, N, 208) - [event_refined, time_refined, boundary_emb]
            event_refined: (B, N, 96)
            time_refined: (B, N, 96)
            prompt_weights: (B, N, K)
            true_length: (B, N) - ground truth lengths
        """
        # Encode events
        event_latent = self.event_encoder(
            input_ids, type_ids, dpe_ids,
            event_mask=event_mask
        )  # (B, N, 64)
        
        # Encode time
        time_emb = self.time_encoder(con_time)  # (B, N, 32)
        
        # Pattern discovery: event-time joint modeling
        # Creates event-level mask from token-level mask
        if event_mask is not None:
            event_level_mask = (event_mask.sum(dim=-1) > 0).float()  # (B, N)
        else:
            event_level_mask = None
        
        event_refined, time_refined, prompt_weights = self.pattern_prompts(
            event_latent, time_emb,
            mask=event_level_mask
        )  # (B, N, 96), (B, N, 96), (B, N, K)
        
        # Compute true length (number of valid tokens per event)
        if event_mask is not None:
            true_length = event_mask.sum(dim=-1).long()  # (B, N)
        else:
            # Fallback: assume all non-zero tokens are valid
            true_length = (input_ids > 0).sum(dim=-1).long()  # (B, N)
        
        # Embed boundary
        boundary_emb = self.boundary_embedding(true_length)  # (B, N, 16)
        
        # Combine into joint latent
        joint_latent = torch.cat([
            event_refined,
            time_refined,
            boundary_emb
        ], dim=-1)  # (B, N, 208)
        
        return joint_latent, event_refined, time_refined, prompt_weights, true_length
    
    def decode(self, event_latent, boundary_mask=None, return_logits=False):
        """
        Decode event latent back to tokens
        
        Args:
            event_latent: (B, N, 96) - refined event latents
            boundary_mask: (B, N, L) - boundary constraint mask (optional)
            return_logits: Whether to return logits
        
        Returns:
            dict with keys 'token', 'type', 'dpe'
        """
        return self.event_decoder(
            event_latent,
            boundary_mask=boundary_mask,
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
    
    def predict_boundary(self, event_latent, prompt_weights):
        """
        Predict boundary (length) distribution
        
        Args:
            event_latent: (B, N, 96) - refined event latents
            prompt_weights: (B, N, K) - pattern weights
        
        Returns:
            length_logits: (B, N, max_len+1)
            length_dist: (B, N, max_len+1)
        """
        prompts = self.pattern_prompts.prompts  # (K, 96)
        return self.boundary_predictor(event_latent, prompt_weights, prompts)
    
    def decode_joint_latent(self, joint_latent, return_logits=False, 
                           deterministic_boundary=True):
        """
        Decode joint latent to events, time, and boundary
        
        Args:
            joint_latent: (B, N, 208) - denoised joint latent
            return_logits: Whether to return logits for events
            deterministic_boundary: Use argmax for boundary (vs sampling)
        
        Returns:
            decoded_events: dict with 'token', 'type', 'dpe'
            decoded_time: (B, N, 1)
            predicted_length: (B, N)
            boundary_mask: (B, N, L)
        """
        # Split joint latent
        event_latent = joint_latent[..., :self.pattern_dim]  # (B, N, 96)
        time_latent = joint_latent[..., self.pattern_dim:2*self.pattern_dim]  # (B, N, 96)
        # boundary_latent not used directly in decoding
        
        # Predict boundary
        # Need prompt_weights, but we don't have them in generation
        # Solution: Re-compute attention to prompts
        with torch.no_grad():
            prompts = self.pattern_prompts.prompts  # (K, 96)
            # Compute similarity as proxy for attention weights
            # event_latent: (B, N, 96), prompts: (K, 96)
            similarity = torch.matmul(
                event_latent,
                prompts.t()
            )  # (B, N, K)
            prompt_weights = torch.softmax(similarity, dim=-1)
        
        length_logits, length_dist = self.predict_boundary(
            event_latent, prompt_weights
        )
        
        # Sample or argmax length
        predicted_length = self.boundary_predictor.sample_length(
            length_dist,
            deterministic=deterministic_boundary
        )  # (B, N)
        
        # Create boundary mask
        B, N = predicted_length.shape
        L = self.event_decoder.max_token_len
        positions = torch.arange(L, device=predicted_length.device).unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        boundary_mask = (positions < predicted_length.unsqueeze(-1)).float()  # (B, N, L)
        
        # Decode events with boundary constraint
        decoded_events = self.event_decoder(
            event_latent,
            boundary_mask=boundary_mask,
            return_logits=return_logits
        )
        
        # Decode time
        decoded_time = self.time_decoder(time_latent)
        
        return decoded_events, decoded_time, predicted_length, boundary_mask
    
    def forward(self, input_ids, type_ids, dpe_ids, con_time, demographics=None, mask=None):
        """
        Full forward pass (for debugging/validation, not used in training)
        
        Args:
            input_ids: (B, N, L)
            type_ids: (B, N, L)
            dpe_ids: (B, N, L)
            con_time: (B, N, 1)
            demographics: (B, D) - not used here
            mask: (B, N) - event mask
        
        Returns:
            reconstructed: dict with decoded outputs
        """
        # Token-level mask
        token_mask = (input_ids > 0).float()
        
        # Encode
        joint_latent, event_refined, time_refined, prompt_weights, true_length = self.encode(
            input_ids, type_ids, dpe_ids, con_time,
            event_mask=token_mask
        )
        
        # Decode
        decoded_events, decoded_time, predicted_length, boundary_mask = self.decode_joint_latent(
            joint_latent,
            return_logits=False,
            deterministic_boundary=True
        )
        
        return {
            'token': decoded_events['token'],
            'type': decoded_events['type'],
            'dpe': decoded_events['dpe'],
            'time': decoded_time,
            'predicted_length': predicted_length,
            'true_length': true_length
        }