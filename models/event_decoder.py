"""
Event Decoder: Latent -> Structured Tokens
Reconstructs (token, type, dpe) from event latents with validity-based length control
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class EventDecoder(nn.Module):
    """
    Decodes event latent vectors back to structured token sequences
    
    Architecture:
        1. Expand latent to hidden dimension
        2. Generate token-level features
        3. Predict token/type/dpe distributions + validity scores
    
    Args:
        event_dim: Input event latent dimension
        hidden_dim: Hidden layer dimension
        vocab_sizes: Dict of vocabulary sizes for each channel
        max_token_len: Maximum tokens per event
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        event_dim: int = 96,
        hidden_dim: int = 256,
        vocab_sizes: dict = None,
        max_token_len: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if vocab_sizes is None:
            vocab_sizes = {'token': 2385, 'type': 7, 'dpe': 15}
        
        self.event_dim = event_dim
        self.hidden_dim = hidden_dim
        self.vocab_sizes = vocab_sizes
        self.max_token_len = max_token_len

        # Project event latent to hidden dimension
        self.latent_proj = nn.Sequential(
            nn.Linear(event_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Expand latent to token-level dimension
        self.token_expander = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim * max_token_len)
        )

        # Transformer for token-level refinement
        self.token_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output heads for each channel
        self.token_head = nn.Linear(hidden_dim, vocab_sizes['token'])
        self.type_head = nn.Linear(hidden_dim, vocab_sizes['type'])
        self.dpe_head = nn.Linear(hidden_dim, vocab_sizes['dpe'])

        # Validity score head
        self.validity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()            
        )

        # Initialize weights                    
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def forward(self, event_latents, return_logits=False):
        """
        Args:
            event_latents: (B, N, event_dim) - event latent vectors
            return_logits: Whether to return raw logits or sampled IDs
        
        Returns:
            If return_logits=True:
                dict with keys 'token', 'type', 'dpe', 'validity'
                - token/type/dpe: (B, N, L, vocab_size) - logits
                - validity: (B, N, L) - validity scores
            Else:
                dict with keys 'token', 'type', 'dpe'
                - Each: (B, N, L) - sampled IDs with padding=0
                - Tokens are masked by validity scores (validity > 0.5)
        """
        B, N, _ = event_latents.shape
        L = self.max_token_len
        
        # Project event latents
        event_h = self.latent_proj(event_latents)   # (B, N, hidden_dim)
        x_expanded = self.token_expander(event_h)   # (B, N, hidden_dim * L)
        x_expanded = rearrange(x_expanded, 'b n (l d) -> b n l d', l=L)   # (B, N, L, hidden_dim)
        x_flat = rearrange(x_expanded, 'b n l d -> (b n) l d')   # (B*N, L, hidden_dim)
        x_refined = self.token_transformer(x_flat)   # (B*N, L, hidden_dim)
        x_refined = rearrange(x_refined, '(b n) l d -> b n l d', b=B, n=N)   # (B, N, L, hidden_dim)

        token_logits = self.token_head(x_refined)
        type_logits = self.type_head(x_refined)
        dpe_logits = self.dpe_head(x_refined)
        # Predict validity scores
        validity_scores = self.validity_head(x_refined).squeeze(-1)
        
        if return_logits:
            return {
                'token': token_logits,
                'type': type_logits,
                'dpe': dpe_logits,
                'validity': validity_scores
            }
        else:
            # Sample IDs (greedy decoding)
            token_ids = token_logits.argmax(dim=-1)  # (B, N, L)
            type_ids = type_logits.argmax(dim=-1)
            dpe_ids = dpe_logits.argmax(dim=-1)
            
            # Mask invalid tokens using validity scores
            validity_mask = (validity_scores > 0.5).float()
            token_ids = token_ids * validity_mask.long()
            type_ids = type_ids * validity_mask.long()
            dpe_ids = dpe_ids * validity_mask.long()
            
            return {
                'token': token_ids,
                'type': type_ids,
                'dpe': dpe_ids
            }
    
    def compute_validity_loss(self, event_latents, target_tokens, mask=None):
        """
        Compute validity loss only (no reconstruction loss for token/type/dpe)
        
        Args:
            event_latents: (B, N, event_dim)
            target_tokens: (B, N, L) - ground truth token IDs (used to compute true validity)
            mask: (B, N, L) - valid token mask (1 for valid, 0 for padding)
        
        Returns:
            loss: Scalar validity loss
            loss_dict: Dictionary of loss components
        """
        logits = self.forward(event_latents, return_logits=True)

        B, N, L = target_tokens.shape
        
        if mask is None:
            mask = (target_tokens > 0).float()
        
        # True validity: 1 for valid tokens, 0 for padding
        true_validity = (target_tokens > 0).float()
        
        # Apply mask if provided
        if mask is not None:
            true_validity = true_validity * mask
        
        # Compute focal loss for validity
        validity_bce = F.binary_cross_entropy(
            logits['validity'],
            true_validity,
            reduction='none'
        )

        # Focal weight: focus on hard examples
        pt = torch.where(
            true_validity == 1,
            logits['validity'],
            torch.ones_like(logits['validity']) - logits['validity']
        )

        focal_weight = (1 - pt) ** 2
        validity_loss = (focal_weight * validity_bce).mean()
        
        loss_dict = {
            'validity': validity_loss.item()
        }
        return validity_loss, loss_dict